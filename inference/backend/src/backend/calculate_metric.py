from collections import defaultdict

from qa_metrics.pedant import PEDANT

pedant = PEDANT()


def answer_equal(gold, predict, question):
    score = pedant.get_score(gold, predict, question)
    return 1 if score > 0.3 else 0


def get_model_buzzpoint(all_run_outputs: list[dict]):
    not_found_index = -1
    default_correctness = 0

    run_outputs_by_qid = defaultdict(list)
    for item in all_run_outputs:
        qid = item["qid"]
        if item["buzz"]:
            run_outputs_by_qid[qid].append(item)

    model_buzzpoint = {}
    for qid in run_outputs_by_qid:
        run_outputs_by_qid[qid].sort(key=lambda x: x["token_position"])
        outputs = run_outputs_by_qid[qid]
        item = outputs[0] if outputs else None
        if item:
            gold = item["answer_primary"]
            predict = item["guess"]
            run_context = item["run_context"]
            correctness = answer_equal(gold, predict, run_context)
            model_buzzpoint[qid] = (item["token_position"], correctness)
        else:
            model_buzzpoint[qid] = (not_found_index, default_correctness)
    return model_buzzpoint


def calculate_buzz_accuracy(model_run_outputs: list[dict]):
    tot_buzz = 0
    correct_buzz = 0
    for item in model_run_outputs:
        gold = item["answer_primary"]
        predict = item["guess"]
        # Note: does it mean we only care about buzzing or not rather than the confidence score?
        if item["buzz"]:
            tot_buzz += 1
            correctness = answer_equal(gold, predict, item["run_context"])
            if correctness == 1:
                correct_buzz += 1
    return correct_buzz / tot_buzz if tot_buzz > 0 else 0.0


def calculate_win_rate_human(model_run_outputs: list[dict], dataset):
    model_buzzpoints = get_model_buzzpoint(model_run_outputs)
    win_rate_human_sum = 0
    for ques_item in dataset:
        qid = ques_item["qid"]
        h_buzz_positions = ques_item["metadata"]["human_buzz_positions"]
        m_buzz_position, is_correct = model_buzzpoints[qid]
        if is_correct:  # only count when model buzzed correctly
            human_correct_team_cnt = 0
            for h_buzz_position, h_buzz_correctness in h_buzz_positions:
                h_buzz_position = max(h_buzz_position - 1, 0)
                # count how many teams buzzed correctly before the model
                if h_buzz_correctness > 0 and h_buzz_position < m_buzz_position:
                    human_correct_team_cnt += 1
            win_rate_human = 1 - (human_correct_team_cnt / len(h_buzz_positions))
            win_rate_human_sum += win_rate_human
    return win_rate_human_sum / len(dataset)


def calculate_win_rate_model(curr_model_run_outputs, other_models_run_outputs):
    curr_buzzpoints = get_model_buzzpoint(curr_model_run_outputs)
    other_model_buzz_positions = {qid: [] for qid in curr_buzzpoints}

    if len(other_models_run_outputs) == 0:
        return 1.0  # No other models to compare to

    for model_run_outputs in other_models_run_outputs.values():
        model_buzzpoints = get_model_buzzpoint(model_run_outputs)
        for qid in model_buzzpoints:
            other_model_buzz_positions[qid].append(model_buzzpoints[qid])

    win_rate_model_sum = 0
    for qid in curr_buzzpoints:
        if curr_buzzpoints[qid][-1]:
            model_correct_team_cnt = 0
            for model_buzz_position, model_buzz_correctness in other_model_buzz_positions[qid]:
                if model_buzz_correctness > 0 and model_buzz_position < curr_buzzpoints[qid][0]:
                    model_correct_team_cnt += 1
            win_rate_model = 1 - (model_correct_team_cnt / len(other_model_buzz_positions[qid]))
            win_rate_model_sum += win_rate_model
    return win_rate_model_sum / len(curr_buzzpoints)
