from __future__ import annotations
import copy
import json
from urllib.error import HTTPError, URLError

import pytest

from expert_agent.agents.llm_assessment import DeepSeekAssessmentAgent, observation_payload
from expert_agent.core.evaluation import evaluate_catalogue, grade_metrics
from expert_agent.core.llm_client import DeepSeekClient, LLMConfig, LLMServiceError
from expert_agent.core.state import ExpertState
from expert_agent.core.workflow import ExpertWorkflow
from expert_agent.schemas.llm_reasoning import CitedEvidence, HypothesisDossier, LLMVote
from expert_agent.tools.evidence_verification import resolve_pointer, verify_evidence
from tests.test_reasoning import make_observation


def cited(value=True, ref='/data_quality/usable'):
    return {'statement': '数据质量测量', 'measurements': [
        {'observation_ref': ref, 'observed_value': value}]}


def dossier():
    observation = make_observation()
    return {'structure_reviews': [
        {'structure_id': structure.id, 'interpretation': 'ambiguous',
         'evidence': [cited(structure.integrated_snr, f'/structures/{structure.id}/integrated_snr')]}
        for structure in observation.structures], 'hypotheses': [
        {'id': 'H_lens', 'family': 'lens', 'configuration': '切向弧', 'status': 'possible',
         'supporting_evidence': [cited()], 'counter_evidence': [],
         'expected_but_unmeasured': ['透镜方程检验']},
        {'id': 'H_disc', 'family': 'non_lens', 'configuration': '前景盘残差', 'status': 'unresolved',
         'supporting_evidence': [], 'counter_evidence': [cited()],
         'expected_but_unmeasured': ['盘模型']}], 'missing_information': ['红移']}


def vote(category='B', value=True):
    return {'vote_category': category,
            'disposition': {'A':'accept', 'B':'accept', 'C':'defer', 'X':'reject', 'insufficient_data':'defer'}[category],
            'rationale': '基于测量的暂定判断', 'supporting_evidence': [cited(value)],
            'limitations': ['未经 Q1 标定'], 'competing_explanations': ['前景残差待检验'],
            'preferred_hypothesis_ids': ['H_disc' if category == 'X' else 'H_lens']}


class FakeClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def complete(self, system, payload):
        self.calls.append((system, copy.deepcopy(payload)))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return json.dumps(response), {'model': 'test', 'request_id': str(len(self.calls)),
                                      'usage': {'total_tokens': 100}}


def assessed_state(usable=True):
    return ExpertWorkflow().run(ExpertState(object_id='test',
                                observation_document=make_observation(usable=usable)))


def test_two_stages_have_no_baseline_grade_or_identity():
    state = assessed_state()
    state.observation_document.input_path = '/data/A/secret-target.fits'
    client = FakeClient([dossier(), vote()])
    result = DeepSeekAssessmentAgent(LLMConfig(backend='deepseek'), client).run(state)
    assert result.llm_assessment['status'] == 'completed'
    assert result.llm_assessment['assessment']['vote_category'] == 'B'
    assert result.llm_assessment['verification']['citations_checked'] == 1
    assert len(client.calls) == 2
    for _, payload in client.calls:
        serialized = json.dumps(payload)
        assert 'secret-target' not in serialized
        assert 'expert_assessment' not in serialized
        assert 'vote_category' not in payload['observation']
    assert 'executable_verification' in client.calls[1][1]


@pytest.mark.parametrize('changed', [False, 1, None, 'true'])
def test_fabricated_or_wrong_type_cited_value_is_rejected(changed):
    evidence = CitedEvidence.model_validate(cited(changed))
    with pytest.raises(ValueError):
        verify_evidence([evidence], {'data_quality': {'usable': True}})


@pytest.mark.parametrize('reference', ['/missing', '/a/-1', '/a/01', '/a/8', '/null', '/bad~3'])
def test_invalid_or_missing_reference_is_rejected(reference):
    with pytest.raises((ValueError, KeyError, IndexError)):
        resolve_pointer({'a': [2], 'null': None}, reference)


def test_false_zero_and_empty_measurements():
    assert resolve_pointer({'f': False}, '/f') is False
    assert resolve_pointer({'z': 0}, '/z') == 0
    assert resolve_pointer({'a/b': {'~x': 2}}, '/a~1b/~0x') == 2
    for value in (None, [], {}, ''):
        with pytest.raises(ValueError):
            resolve_pointer({'x': value}, '/x')


def test_machine_precision_rounding_is_allowed_but_changed_measurement_is_not():
    document = {'data_quality': {'usable': 82.67562963182148}}
    verify_evidence([CitedEvidence.model_validate(cited(82.6756296318215))], document)
    with pytest.raises(ValueError):
        verify_evidence([CitedEvidence.model_validate(cited(82.676))], document)


def test_invalid_hypothesis_stops_before_second_api_call():
    invalid = dossier()
    invalid['hypotheses'][0]['supporting_evidence'] = [cited(False)]
    client = FakeClient([invalid, vote()])
    result = DeepSeekAssessmentAgent(LLMConfig(backend='deepseek', max_repairs=0), client).run(assessed_state())
    assert result.llm_assessment['status'] == 'failed'
    assert result.llm_assessment['assessment'] is None
    assert len(client.calls) == 1


def test_one_bounded_repair_corrects_citations_before_adjudication():
    invalid = dossier()
    invalid['hypotheses'][0]['supporting_evidence'] = [cited(False)]
    client = FakeClient([invalid, dossier(), vote()])
    result = DeepSeekAssessmentAgent(LLMConfig(backend='deepseek'), client).run(assessed_state())
    assert result.llm_assessment['status'] == 'completed'
    assert len(client.calls) == 3
    assert client.calls[1][1]['verification_error']['actual_value'] is True
    assert result.llm_assessment['stages'][0]['verification']['passed'] is False


def test_failed_repair_cannot_loop_indefinitely():
    invalid = dossier()
    invalid['hypotheses'][0]['supporting_evidence'] = [cited(False)]
    client = FakeClient([invalid, invalid])
    result = DeepSeekAssessmentAgent(LLMConfig(backend='deepseek'), client).run(assessed_state())
    assert result.llm_assessment['status'] == 'failed'
    assert len(client.calls) == 2


def test_ignoring_a_measured_structure_is_rejected():
    invalid = dossier()
    invalid['structure_reviews'] = invalid['structure_reviews'][:1]
    client = FakeClient([invalid])
    result = DeepSeekAssessmentAgent(LLMConfig(backend='deepseek', max_repairs=0), client).run(assessed_state())
    assert result.llm_assessment['status'] == 'failed'
    assert result.llm_assessment['stages'][0]['verification']['error'] == 'incomplete_structure_review'


def test_missing_structure_measurement_fails_ablation_check():
    state = assessed_state()
    state.observation_document.structures[0].tangential_misalignment_deg = None
    # A null value cannot support the unchanged hypothesis evidence.
    hyp = dossier()
    sid = state.observation_document.structures[0].id
    hyp['structure_reviews'][0]['evidence'] = [cited(10.0, f'/structures/{sid}/tangential_misalignment_deg')]
    client = FakeClient([hyp])
    result = DeepSeekAssessmentAgent(LLMConfig(backend='deepseek', max_repairs=0), client).run(state)
    assert result.llm_assessment['status'] == 'failed'


def test_multiband_measurements_remain_in_anonymous_handoff():
    state = assessed_state()
    state.observation_document.supplementary_diagnostics['multiband'] = {
        'status': 'measured', 'bands': {'VIS': {'available': True}, 'NIR_Y': {'available': True}},
        'input_sha256': 'identity', 'relationship_audit': {'grade_use': 'not a label', 'component_pairs': []}}
    payload = observation_payload(state)
    assert payload['multiband']['bands']['NIR_Y']['available']
    assert 'input_sha256' not in payload['multiband']
    assert 'grade_use' not in payload['relationships']


def test_model_references_are_materialized_by_tools():
    evidence = CitedEvidence.model_validate({'statement': '测量', 'measurements': [
        {'observation_ref': '/x'}]})
    verify_evidence([evidence], {'x': 1.23456789})
    assert evidence.measurements[0].observed_value == 1.23456789


def test_unreferenced_extraction_measurements_are_not_current_evidence():
    state = assessed_state()
    from expert_agent.schemas.observation import Measurement
    state.observation_document.measurements.append(Measurement(
        id='stale_scene', kind='structure_count', value=99, reliability=1,
        method='before_merge', description='old scene', correlation_group='scene'))
    payload = observation_payload(state)
    assert 'stale_scene' not in payload['measurements']


def test_non_lens_alternative_is_required():
    invalid = dossier()
    invalid['hypotheses'][1]['family'] = 'lens'
    with pytest.raises(ValueError):
        HypothesisDossier.model_validate(invalid)


def test_unknown_hypothesis_is_rejected():
    invalid = vote()
    invalid['preferred_hypothesis_ids'] = ['invented']
    result = DeepSeekAssessmentAgent(LLMConfig(backend='deepseek', max_repairs=0),
                                     FakeClient([dossier(), invalid])).run(assessed_state())
    assert result.llm_assessment['status'] == 'failed'


def test_unusable_data_cannot_receive_a_grade():
    hyp = dossier()
    for item in hyp['hypotheses']:
        for key in ('supporting_evidence', 'counter_evidence'):
            for evidence in item[key]:
                evidence['measurements'][0]['observed_value'] = False
    result = DeepSeekAssessmentAgent(LLMConfig(backend='deepseek', max_repairs=0),
                                     FakeClient([hyp, vote('A', False)])).run(assessed_state(False))
    assert result.llm_assessment['status'] == 'failed'


def test_adjudication_repair_uses_one_shared_budget():
    invalid = vote()
    invalid['preferred_hypothesis_ids'] = ['invented']
    client = FakeClient([dossier(), invalid, vote()])
    result = DeepSeekAssessmentAgent(LLMConfig(backend='deepseek'), client).run(assessed_state())
    assert result.llm_assessment['status'] == 'completed'
    assert len(client.calls) == 3
    assert result.llm_assessment['stages'][1]['verification']['error'] == 'unknown_preferred_hypothesis'


def test_hypothesis_and_adjudication_share_repair_budget():
    invalid_dossier = dossier()
    invalid_dossier['hypotheses'][0]['supporting_evidence'] = [cited(False)]
    invalid_vote = vote()
    invalid_vote['preferred_hypothesis_ids'] = ['invented']
    client = FakeClient([invalid_dossier, dossier(), invalid_vote])
    result = DeepSeekAssessmentAgent(LLMConfig(backend='deepseek'), client).run(assessed_state())
    assert result.llm_assessment['status'] == 'failed'
    assert len(client.calls) == 3


def test_disposition_and_evidence_are_required():
    invalid = vote('A')
    invalid['disposition'] = 'reject'
    with pytest.raises(ValueError):
        LLMVote.model_validate(invalid)
    invalid = vote()
    invalid['supporting_evidence'] = []
    with pytest.raises(ValueError):
        LLMVote.model_validate(invalid)


def test_missing_key_does_not_call_network(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    monkeypatch.setattr('expert_agent.core.llm_client.urlopen', lambda *a, **kw: pytest.fail('network used'))
    with pytest.raises(LLMServiceError, match='missing_api_key'):
        DeepSeekClient(LLMConfig()).complete('json', {})


@pytest.mark.parametrize('failure', [HTTPError('https://api.deepseek.com', 402, 'secret-body', {}, None),
                                    URLError('secret-token')])
def test_provider_error_is_sanitized(monkeypatch, tmp_path, failure):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-secret')
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr('expert_agent.core.llm_client.urlopen', fail)
    with pytest.raises(LLMServiceError) as exc:
        DeepSeekClient(LLMConfig()).complete('json', {})
    assert 'secret' not in str(exc.value)


def test_environment_overrides_local_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / '.env').write_text('DEEPSEEK_MODEL=local\nLENS_AGENT_JUDGMENT=deepseek\n')
    monkeypatch.setenv('DEEPSEEK_MODEL', 'environment')
    monkeypatch.delenv('LENS_AGENT_JUDGMENT', raising=False)
    config = LLMConfig.from_environment()
    assert config.backend == 'deepseek'
    assert config.model == 'environment'
    assert LLMConfig.from_environment(backend='rules').backend == 'rules'


def test_rules_workflow_is_offline_even_when_local_default_is_deepseek(monkeypatch):
    monkeypatch.setenv('LENS_AGENT_JUDGMENT', 'deepseek')
    monkeypatch.setattr(DeepSeekClient, 'complete', lambda *args: pytest.fail('network used'))
    result = ExpertWorkflow().run(ExpertState(observation_document=make_observation()))
    assert result.llm_assessment is None
    assert 'llm_assessment' not in result.to_dict()


def test_failed_llm_preserves_rules_report(monkeypatch):
    def fail(*args):
        raise LLMServiceError('http_402')
    monkeypatch.setattr(DeepSeekClient, 'complete', fail)
    result = ExpertWorkflow(llm_config=LLMConfig(backend='deepseek')).run(
        ExpertState(object_id='test', observation_document=make_observation()))
    assert result.decision_audit.audit_passed
    assert result.final_decision['llm_grade'] is None
    assert result.final_decision['llm_assessment']['error'] == 'http_402'
    assert '没有可用的模型等级' in result.final_report_markdown


def test_metrics_keep_abstentions_in_denominator():
    metrics = grade_metrics([('A', 'A'), ('A', 'insufficient_data'), ('B', 'unavailable'), ('B', 'A')])
    assert metrics['coverage'] == .5
    assert metrics['accuracy_all_targets'] == .25
    assert metrics['accuracy_definite_only'] == .5
    assert metrics['recall_by_grade']['B'] == 0


def test_catalogue_missing_result_is_not_silently_dropped(tmp_path):
    csv = tmp_path / 'reference.csv'
    csv.write_text('id_str,grade\none,A\ntwo,B\n')
    root = tmp_path / 'results'
    root.mkdir()
    metrics = evaluate_catalogue(csv, root)
    assert metrics['deepseek']['n'] == 2
    assert metrics['deepseek']['accuracy_all_targets'] == 0
    assert all(item['llm_error'] == 'missing_result' for item in metrics['targets'])
