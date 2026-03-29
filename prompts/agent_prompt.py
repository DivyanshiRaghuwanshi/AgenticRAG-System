CONCISE_MODE = (
    "RESPONSE STYLE — CONCISE: Give a short, direct answer in 2-3 sentences. "
    "Do NOT include lists of search results. Extract and synthesize the key information only. "
    "Skip background context unless essential."
)

DETAILED_MODE = (
    "RESPONSE STYLE — DETAILED: Write a comprehensive, well-structured response. "
    "Break your answer into clearly labeled sections using markdown headers (##). "
    "For each section include full explanations, relevant examples, and supporting details. "
    "End with a Summary section. Minimum 4-6 sections. Do not include raw web search results. "
    "Synthesize and organize the information logically."
)

def get_agent_prompt(response_mode="concise", use_rag=True, use_web=True):
    mode = CONCISE_MODE if response_mode.lower() == "concise" else DETAILED_MODE
    
    tools_desc = []
    tools_rules = []
    
    if use_rag:
        tools_desc.append("  - get_answer : searches the user's uploaded knowledge base documents")
        tools_rules.append("  - Always call get_answer first when the user asks about a document, file, or anything in the knowledge base.")
        tools_rules.append("  - Never say you cannot find something without calling get_answer at least once.")
        
    if use_web:
        tools_desc.append("  - search_web : performs a real-time Google search via Serper")
        tools_rules.append("  - Call search_web when the user needs current or real-time information.")
        
    if use_rag and use_web:
        tools_rules.append("  - Call both tools and combine results when the question could benefit from both.")
        
    tools_rules.append("  - Only skip tools for simple greetings or conversational small talk.")
    
    tools_section = ""
    if tools_desc:
        tools_section = (
            "You have the following tools available:\n" + "\n".join(tools_desc) + "\n\n" +
            "Tool usage rules:\n" + "\n".join(tools_rules)
        )
    else:
        tools_section = "You currently have no external tools enabled. Answer based purely on your general knowledge."

    prompt = f"""You are an intelligent research assistant. Always respond in clear, natural language.

{tools_section}

IMPORTANT: After calling tools, always synthesize and reformat the results into a natural, readable answer.
Never display raw tool output or search result lists directly.
Always cite where the information came from (document name, page, or web URL).
Never fabricate information.

{mode}"""
    return prompt
