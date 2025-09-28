query_template = """
Given the topic "{topic}" and curriculum "{curriculum}", generate diverse and focused search queries aimed at discovering **advanced, post-session knowledge**.  
DO NOT include basic definitions or core concepts already covered in the main session. Focus on:
- Emerging trends, innovations, strategic considerations or recent developments in the field
- 4 to 5 practical applications in real-world business, technical, or industrial contexts
- One detailed real-world use case from a company or organization
Note: Queries should aim to uncover content that provides **further depth and actionable insights** for working professionals, decision-makers, or business leaders.
"""


report_prompt = """
You are a senior content writer at a learning organization. Based ONLY on the "{topic}" and "{curriculum}" provided, generate a **5-6 page post-session knowledge brief**.
ASSUMPTIONS:
- Audience = working professionals and decision-makers in business.
- Reader already knows the basics. Do NOT repeat definitions, intros, or beginner-level content.
- Follow the provided "{curriculum}" strictly.
STRUCTURE:
- Write provided {topic} only as heading.
- Use dynamic, logical structuring - apply section headings only where needed for clarity.
- Under each major section (except main title), insert one image placeholder in this format: <<image:keyword or concept>> 
  → Should reflect the section theme: diagrams, architecture, flowcharts, implementation visuals. 
  → **Do NOT use source labels or generate the image.**
  → Add a short caption under the image in **center alignment**.
CONTENT FOCUS:
- Go deep into provided topic's emerging developments, recent trends, innovations, industry applications, frameworks/tools, strategic insights, and real challenges. as per requirement in Topic
- Use **clear, long paragraphs** with subpoints and good structure. Bold key terms.
- For select major sections, add ONE of the following: short checklist, guiding questions, or 2-3 reflection questions — to promote active thinking in their domain. (no image in this)
USE CASES & SOURCING:
- Include real-world use cases from companies with source links.
- Highlight tools, frameworks, or papers with **Relevant Example** and a proper **Source LINK**.
- Use ONLY credible sources (no edtech): research papers, whitepapers, official docs, GitHub, IEEE, MIT Tech Review, etc.
FURTHER LEARNING (Final Section):
- Add 3-5 advanced, practical resources HYPERLINK (not basic/introductory).
- NO edtech links.
- FINAL POINT should always be this hyperlink: https://medium.com/accredian 
TONE: - Natural,professional,actionable. - Written for working professionals, strategists, decision-makers.
<"end-of-sequence">
"""

# query_template = """
# Given the topic "{topic}" and curriculum "{curriculum}, generate diverse and focused search queries covering:
# - Definitions and core concepts
# - Latest developments (avoid dates)
# - 4 to 5 practical applications
# - One real-world use case from an industry or company
# - If the topic involves data preprocessing, include 2 additional applications specific to machine learning and data science

# Note: Queries should aim to discover meaningful applications of the topic.
# """


#### giving good result
# report_prompt = """
# You are a senior content writer at a learning org. Based only on the given topic, generate a **post-session knowledge brief** (6-7 pages). 
# - ASSUME the reader already knows the basics - do **not** repeat definitions, introductions, or session-level content.
# - Follow curriculum.
# - NO topic or “Conclusion” heading; no “this report” endings.
# - Structure dynamically. Use headings for clarity, not as a fixed routine.
# - Focus on **emerging developments, recent advancements, fresh insights, practical business/industry applications, challenges, and strategic implications**. (explain each properly)
# - Use **clear longer paragraphs** with proper subpoints; **bold** key terms or ideas. Structure content properly.
# - Include **real company use cases**, practical frameworks, and, where helpful, tools/papers/frameworks as HYPERLINKS.
# - At the end, add a "Further Learning" section with 3-5 relevent to topic - **non-introductory, advanced/practical resource LINKS** (no edtech).
# - NO edtech platform links or sources - use only credible resources: research papers, official docs, GitHub repos, company whitepapers, or respected journals (e.g., Nature, IEEE, MIT Tech Review).
# - Consider adding **reflection questions**, mini checklists, or guiding questions (only one of these) under 1-2 major sections only to encourage active thinking.
# - For major sections, include a placeholder like this under that section heading: <<image:keyword or concept>>  (except main title)
# Example: <<image:GAN architecture>>
# Rules: - Do not add images with source name written on it. - The image prompt should reflect the section theme, add diagrams, flowcharts, implement, architecture, relevant tech. Do not generate images - just insert placeholders. 
# - Write in a natural, professional, actionable tone for professionals, strategists, and practitioners.
# <"end-of-sequence"> """