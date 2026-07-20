from src.expert_agent.core.state import ExpertState
from src.expert_agent.core.workflow import ExpertWorkflow

def main():

    state = ExpertState(
        image_path="example_image.jpg",
        object_id="test_object"
    )

    workflow = ExpertWorkflow()

    result = workflow.run(state)

    print("\n========== OBSERVATIONS ==========")
    print(result.observations)

    print("\n========== HYPOTHESES ==========")
    print(result.hypotheses)

    print("\n========== EVIDENCE ==========")
    print(result.evidence)

    print("\n========== CONTRADICTIONS ==========")
    print(result.contradictions)

    print("\n========== AUDIT ==========")
    print(result.audit_result)

    print("\n========== FINAL REPORT ==========")
    print(result.final_decision)


if __name__ == "__main__":
    main()