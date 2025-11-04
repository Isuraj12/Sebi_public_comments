import pandas as pd 
import numpy as np 
import streamlit as st 
import google.generativeai as genai 
import sqlite3 
import os

# --- Configure Gemini ---
genai.configure(api_key="AIzaSyBfODz4dMqxnSNJBojgEAL38SyUXNCv3vs")
model = genai.GenerativeModel("gemini-2.0-flash")

# --- Function to Load Data ---
def get_data():
    conn = sqlite3.connect('sebi_circulars.db')
    df = pd.read_sql('SELECT * FROM circulars', conn)
    conn.close()
    return df

# --- Helper to parse mixed date formats ---
def parse_dates(series):
    # Try multiple known formats (SEBI website uses various)
    formats = [
        "%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d %b %Y", "%d %B %Y",
        "%b %d, %Y", "%B %d, %Y"
    ]
    for fmt in formats:
        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
        if parsed.notna().sum() > 0:
            return parsed
    # fallback: try generic fuzzy parsing
    return pd.to_datetime(series, errors="coerce", dayfirst=True)

# --- Streamlit App ---
st.title(" SEBI Draft Circular Analysis")

df = get_data()

if df.empty:
    st.warning("No data available.")
else:
    if "Date" not in df.columns or "Title" not in df.columns:
        st.error("The 'Date' or 'Title' column is missing from the data.")
    else:
        # --- Robust Date Conversion ---
        df["Date"] = parse_dates(df["Date"])
        valid_dates = df["Date"].dropna()

        if valid_dates.empty:
            st.error(" Could not parse any valid dates. Check your 'Date' column format in the database.")
            st.dataframe(df.head())  # show a preview for debugging
        else:
            # --- Filter Section ---
            st.sidebar.header(" Filter Circulars")

            # Title filter
            title_filter = st.sidebar.text_input("Search by Title (keywords):", "")

            # Safe handling of min/max dates
            min_date = valid_dates.min().date()
            max_date = valid_dates.max().date()

            # Date range filter
            start_date, end_date = st.sidebar.date_input(
                "Filter by Date Range:",
                [min_date, max_date],
                min_value=min_date,
                max_value=max_date
            )

            # Apply filters
            filtered_df = df.copy()

            if title_filter:
                filtered_df = filtered_df[
                    filtered_df["Title"].str.contains(title_filter, case=False, na=False)
                ]

            if isinstance(start_date, list) or isinstance(start_date, tuple):
                start_date, end_date = start_date[0], start_date[1]

            filtered_df = filtered_df[
                (filtered_df["Date"] >= pd.to_datetime(start_date)) &
                (filtered_df["Date"] <= pd.to_datetime(end_date))
            ]

            # Display filtered options
            if filtered_df.empty:
                st.warning("No circulars found for the selected filters.")
            else:
                filtered_df["Display"] = filtered_df["Date"].dt.strftime("%d %b %Y") + " - " + filtered_df["Title"].astype(str)

                selected_circular = st.selectbox("Select a Circular:", filtered_df["Display"].tolist())

                if selected_circular:
                    selected_row = filtered_df[filtered_df["Display"] == selected_circular].iloc[0]

                    st.subheader("Circular Details")
                    if pd.notna(selected_row['Date']):
                        st.write(f"**Date:** {selected_row['Date'].strftime('%d %b %Y')}")
                    else:
                        st.write("**Date:** Not Available")
                    st.write(f"**Title:** {selected_row['Title']}")

                    if "PDF_URL" in df.columns and pd.notna(selected_row["PDF_URL"]):
                        st.markdown(f"[View PDF]({selected_row['PDF_URL']})")

                    if "Extracted_Text" in df.columns:
                        circular_text = selected_row["Extracted_Text"]

                        st.subheader("Extracted Text")
                        st.text_area("Extracted Text:", circular_text, height=300)

                        
                        if st.button(" Generate Summary"):
                            with st.spinner("Generating Summary..."):
                                prompt = f"""
You are a financial regulatory analyst with expertise in SEBI regulations, securities law, and capital-market policy.
The user will provide the extracted text of a SEBI Consultation Paper or Draft Circular.
Your task is to produce a structured review document in the format used by legal policy and market-regulation think tanks.

The output must follow the structure and depth below. Write in a formal, analytical, and neutral tone, formatted cleanly for Word/Markdown export.

🔹 STRUCTURE AND GUIDELINES
Market Classification

Classify the consultation paper under one of the following divisions based on subject matter:

Primary Markets – IPOs, FPOs, REITs, InvITs, public issues, and capital formation.

Secondary Markets – Trading, exchanges, market intermediaries, surveillance, disclosures, and investor protection.

Commodity Markets – Regulation of commodity exchanges and derivatives.

External Markets – Cross-border listings, foreign portfolio investment, GDRs, ADRs, or external capital flows.

Output at the top:
Market Classification: [Primary / Secondary / Commodity / External]

1. Background / Regulatory Context / Introduction

Include:

A clear overview of what the Consultation Paper is about.

The existing regulatory framework under SEBI Regulations, circulars, or Master Circulars.

The evolution and pain points that triggered SEBI’s reform initiative.

Any relevant legislative or circular history, including important amendment dates or regulations.

Purpose of the proposed reform in one or two sentences.

2. Summary of Key Proposals

Provide a neutral summary of what SEBI has proposed — without opinion or interpretation.

Use bullet points or short paragraphs for clarity.

Mention each proposal separately as it appears in the Consultation Paper (e.g., Proposal 1, Proposal 2, etc.).

Avoid legal commentary here — keep it descriptive.

3. Critical Analysis of the Proposals

This is the analytical core of the review.
For each distinct proposal, use the sub-structure below:

Proposal [Number]: [Title of Proposal from CP]

Concept Proposed:
Provide a neutral summary of what SEBI is proposing in this section — exactly as per the Consultation Paper.

SEBI’s Rationale:
Summarize SEBI’s policy reasoning or intent behind the proposal — why it is needed, what problem it solves, and what benefits it seeks to bring.

Global Benchmarking:
Compare this proposal with how similar regulatory issues are handled in three to four relevant international jurisdictions (select from: US SEC, UK FCA, EU ESMA, Singapore MAS, Hong Kong SFC/HKEX).

Identify comparable frameworks or approaches in these countries.

Discuss whether India’s proposal aligns or diverges from them.

Provide references or URLs only for this subsection (e.g., links to regulator websites or policy documents).

Critical Assessment & Recommendations:

Our Stance:
State the team’s position – Accepted / Accepted with Modifications / Not Accepted.

Supporting Rationale:
Provide detailed justification for the stance. Explain the potential regulatory, legal, or market impact (positive or negative).

Proposed Modifications / Safeguards (if applicable):
If accepted with modification, propose specific, actionable alternatives, such as:

Revised thresholds or timelines

Additional disclosure requirements

Transitional or phased implementation clauses

Anti-avoidance or investor-protection safeguards

4. Conclusion and Overall Recommendations

Summarize:

Whether SEBI’s approach is conceptually sound and internationally aligned.

The overall impact on market efficiency, investor protection, and regulatory coherence.

Provide 3–5 overall recommendations on how SEBI could refine or clarify its proposals before finalizing the framework.

Use bullet points for clarity and conciseness.

5. Key Questions for the Ministry of Finance (MoF)

List five critical questions that the Ministry of Finance should ask SEBI about this Consultation Paper.
These should be policy-oriented, forward-looking, and designed to:

Challenge underlying assumptions,

Strengthen implementation logic, or

Enhance alignment with India’s market-development objectives.

Example format:

How will SEBI ensure that [specific reform] does not create duplicative compliance obligations for already listed entities?

What mechanisms are in place to align this reform with global capital-market standards?
(Continue up to 5.)
{circular_text}"""
                                

                                try:
                                    response = model.generate_content(prompt)
                                    summary = response.text
                                    st.success(" Summary Generated!")
                                    st.subheader("Summary")
                                    st.write(summary)

                                except Exception as e:
                                    st.error(f"An error occurred while generating the summary: {e}")
                    else:
                        st.error("The 'Extracted_Text' column is missing from the data.")
