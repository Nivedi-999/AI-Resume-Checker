import re

def calculate_completeness(text):
    sections = ["education", "experience", "skills", "projects", "summary"]
    found = sum(1 for sec in sections if sec in text.lower())
    return round((found / len(sections)) * 100, 2)

def match_skills(text, skill_list):
    found = [s for s in skill_list if re.search(r'\b' + re.escape(s) + r'\b', text.lower())]
    match_score = round((len(found) / len(skill_list)) * 100, 2)
    return match_score, found
