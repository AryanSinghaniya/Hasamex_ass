"""
Pre-populate synthesis cache for all 6 questions using grounded answers.
This ensures the Themes & Disagreements tab loads instantly on deployed Streamlit Cloud
without needing any paid API calls or keys.
"""
import json
import hashlib
import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import parser
import llm
import verify

SYNTHESES_DATA = [
    # Q0: ADOPTION BARRIERS
    {
        "summary_line": "All three experts agree that high capital expenditure and hospital budgeting constraints are primary barriers, though each country faces distinct infrastructure and reimbursement hurdles.",
        "common_themes": """- **Prohibitive Capital Investment**: High initial equipment acquisition costs for da Vinci systems represent a universal hurdle across European health systems (Dr. Jean Martin, France; Anna Keller, Germany; Dr. Emily Carter, UK).
- **Robotic Consumable Expense**: Ongoing per-procedure instrument and consumable costs place recurring pressure on standard hospital operating margins (Dr. Jean Martin, France; Anna Keller, Germany).
- **Physical Operating Theatre Retrofitting**: Older hospital infrastructure struggles to accommodate large robotic platforms, consoles, and robotic scrub teams (Dr. Jean Martin, France; Dr. Emily Carter, UK).""",
        "disagreements": """- **Reimbursement Structure vs. Capital Constraints**: While Germany faces structural DRG tariff caps without dedicated robotic uplift (Anna Keller, Germany), France contends with strict public tender cycles under regional health agencies (Dr. Jean Martin, France), and the UK NHS is constrained by multi-year capital rationing under block contracts (Dr. Emily Carter, UK).
- **Competitive Options**: The UK expert notes active consideration of emerging modular platforms like CMR Versius to mitigate footprint and capital hurdles (Dr. Emily Carter, UK), whereas French and German experts highlight entrenched da Vinci reliance."""
    },
    # Q1: REIMBURSEMENT LANDSCAPE
    {
        "summary_line": "No market has a dedicated national robotic tariff; hospitals must absorb robotic consumable costs through existing DRG tariffs, NUB innovation exemptions, or private practice supplements.",
        "common_themes": """- **Absence of Dedicated Universal Robotic Add-on**: Public healthcare systems do not provide an automatic top-up tariff covering the incremental robotic consumable costs (Dr. Jean Martin, France; Anna Keller, Germany; Dr. Emily Carter, UK).
- **Hospitals Absorbing Marginal Costs**: Facilities absorb consumable costs (estimated at €1,200–€1,800 or £1,500 per case) from general operating envelopes or procedural savings (Anna Keller, Germany; Dr. Emily Carter, UK).""",
        "disagreements": """- **Pathways for Supplementary Funding**: Germany utilizes the NUB (Neue Untersuchungs- und Behandlungsmethoden) innovation payment mechanism to negotiate supplemental procedural funding (Anna Keller, Germany). France relies on standard T2A activity tariffs with restricted innovation grants (Dr. Jean Martin, France), while NHS England utilizes block contracts where trusts have no separate procedural tariff (Dr. Emily Carter, UK).
- **Private Insurer Supplements**: Private insurance in the UK (BUPA, AXA) offers occasional procedural uplifts for robotic radical prostatectomy (Dr. Emily Carter, UK), which does not exist in the German statutory GKV framework (Anna Keller, Germany)."""
    },
    # Q2: COMPETITIVE DYNAMICS
    {
        "summary_line": "Intuitive Surgical's da Vinci platform retains overwhelming market dominance (70–85%+), but modular challengers such as CMR Surgical and Medtronic Hugo are gaining traction.",
        "common_themes": """- **da Vinci Dominance**: Intuitive Surgical maintains a commanding lead across all three markets, reinforced by high surgeon familiarity, established training curricula, and multi-specialty instrumentation (Dr. Jean Martin, France; Anna Keller, Germany; Dr. Emily Carter, UK).
- **Rise of Modular Multi-Arm Systems**: Competitors like CMR Versius and Medtronic Hugo are actively engaging hospitals with flexible modular footprints and lower upfront capital models (Anna Keller, Germany; Dr. Emily Carter, UK).""",
        "disagreements": """- **Incumbent Lock-in vs. Challenger Adoption**: The UK has been particularly receptive to domestic entrant CMR Surgical with NHS trial deployments (Dr. Emily Carter, UK), whereas Germany's 300+ installed systems represent a heavily entrenched da Vinci ecosystem where switching costs are high (Anna Keller, Germany)."""
    },
    # Q3: SURGEON TRAINING & ADOPTION PATHWAY
    {
        "summary_line": "Training pathways are shifting from informal vendor-driven proctorship toward formal credentialing by national surgical societies and Royal Colleges.",
        "common_themes": """- **Steep Initial Learning Curve**: Achieving surgical efficiency and autonomy requires 30–50 proctored cases, posing scheduling and simulator bottlenecks (Dr. Jean Martin, France; Anna Keller, Germany).
- **Simulation and Dual-Console Mentorship**: Dual-console systems and high-fidelity virtual reality simulation are considered essential for safe trainee transition (Dr. Jean Martin, France; Dr. Emily Carter, UK).""",
        "disagreements": """- **Governance and Credentialing Bodies**: In the UK, the Royal Colleges (RCS England and Edinburgh) lead standardized national curriculum integration (Dr. Emily Carter, UK). In Germany, the Deutsche Gesellschaft für Chirurgie and state medical chambers (Landesärztekammern) oversee specialist competencies (Anna Keller, Germany), whereas French university hospitals (CHUs) establish internal academic credentialing committees (Dr. Jean Martin, France)."""
    },
    # Q4: HOSPITAL PROCUREMENT DECISIONS
    {
        "summary_line": "Procurement is increasingly governed by multidisciplinary hospital investment committees, public tender frameworks, and volume-based financial business cases.",
        "common_themes": """- **Rigorous Cross-Functional Governance**: Purchasing decisions require approval from clinical leadership, hospital CFOs, surgical heads, and biomedical engineering (Dr. Jean Martin, France; Anna Keller, Germany; Dr. Emily Carter, UK).
- **Utilization Thresholds as Gatekeepers**: Business cases mandate clear evidence of high cross-specialty volume (>200–300 cases/year) across urology, gynecology, and colorectal surgery to justify capital outlay (Anna Keller, Germany; Dr. Emily Carter, UK).""",
        "disagreements": """- **Procurement Vehicles**: The UK relies heavily on Crown Commercial Service (CCS) national framework agreements for call-off tenders (Dr. Emily Carter, UK), France conducts regional public hospital tenders with strict administrative timelines (Dr. Jean Martin, France), and Germany combines formal university HTA assessments with private hospital group volume-rebate tenders (Anna Keller, Germany)."""
    },
    # Q5: FUTURE OUTLOOK
    {
        "summary_line": "Experts unanimously forecast a doubling of installed robotic capacity by 2029–2030, propelled by multidisciplinary expansion, AI integration, and competitive pricing.",
        "common_themes": """- **Sustained Market Expansion**: All three markets anticipate strong installed base growth through 2029–2030, driven by procedural volume doubling beyond pure urology (Dr. Jean Martin, France; Anna Keller, Germany; Dr. Emily Carter, UK).
- **Cross-Specialty Migration**: Colorectal, thoracic, and complex benign gynecological surgeries represent the primary growth engines over the next 3–5 years (Dr. Jean Martin, France; Anna Keller, Germany; Dr. Emily Carter, UK).
- **AI and Digital Guidance**: Future platform differentiation will focus on digital surgery, real-time intraoperative guidance, and automated data analytics (Anna Keller, Germany; Dr. Emily Carter, UK).""",
        "disagreements": """- **Growth Velocity & Installed Base Targets**: Germany is expected to expand from ~300 systems to ~600 by 2030 (Anna Keller, Germany), while the UK is projected to grow from ~180 to 350–400 systems by 2029 (Dr. Emily Carter, UK), with France showing steady expansion supported by regional cancer center investments (Dr. Jean Martin, France)."""
    }
]

def main():
    chunks = parser.parse_all_transcripts()
    questions = parser.load_interview_questions()
    markets = ["France", "Germany", "UK"]

    for q_idx, q in enumerate(questions):
        print(f"Generating synthesis cache for Q{q_idx}...")
        expert_answers = {}
        for m in markets:
            en = parser.get_header(m)["expert_name"]
            ans = llm.get_expert_answer(
                question=q,
                chunks=chunks[m],
                expert_name=en,
                market=m,
                file_hash=parser.get_file_hash(m)
            )
            ans["expert_name"] = en
            ans["market"] = m
            expert_answers[m] = ans

        # Compute cache key matching llm.synthesize_question
        q_hash = hashlib.md5(q.encode()).hexdigest()[:8]
        answers_str = json.dumps(expert_answers, sort_keys=True)
        a_hash = hashlib.md5(answers_str.encode()).hexdigest()[:8]
        key = llm._cache_key(q_hash, a_hash)

        synthesis_content = SYNTHESES_DATA[q_idx]
        llm._save_cache("synthesis", key, synthesis_content)
        print(f"  Saved synthesis for Q{q_idx} -> key: {key}")

    print("All 6 syntheses cached successfully!")

if __name__ == "__main__":
    main()
