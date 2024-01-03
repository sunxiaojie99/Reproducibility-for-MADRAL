from collections import Counter
import sys
import json


def quality_checks_qids(qids_to_relevant_passageids, qids_to_ranked_candidate_passages):
    """Perform quality checks on the dictionaries
    Args:
    p_qids_to_relevant_passageids (dict): dictionary of query-passage mapping
        Dict as read in with load_reference or load_reference_from_stream
    p_qids_to_ranked_candidate_passages (dict): dictionary of query-passage candidates
    Returns:
        bool,str: Boolean whether allowed, message to be shown in case of a problem
    """
    message = ''
    allowed = True

    # Check that we do not have multiple passages per query
    for qid in qids_to_ranked_candidate_passages:
        # Remove all zeros from the candidates
        duplicate_pids = set(
            [item for item, count in Counter(qids_to_ranked_candidate_passages[qid]).items() if count > 1])

        if len(duplicate_pids - set([0])) > 0:
            message = "Cannot rank a passage multiple times for a single query. QID={qid}, PID={pid}".format(
                qid=qid, pid=list(duplicate_pids)[0])
            allowed = False

    return allowed, message


def compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages):
    """Compute MRR metric 计算mrr分数
    把标准答案在被评价系统给出结果中的排序取倒数作为它的准确度，再对所有的问题取平均
    Args:
    p_qids_to_relevant_passageids (dict): dictionary of query-passage mapping
        Dict as read in with load_reference or load_reference_from_stream
    p_qids_to_ranked_candidate_passages (dict): dictionary of query-passage candidates
    Returns:
        dict: dictionary of metrics {'MRR': <MRR Score>}
    """
    all_scores = {}
    MRR = 0
    macro_recall1 = 0
    macro_recall50 = 0
    macro_recallall = 0
    ranking = []
    ranking_qid_2_idx_dict = {}
    ranking_qid_2_idx_dict_top_10 = {}
    not_find_qid = []
    recall_q_top1 = set()
    recall_q_top50 = set()
    recall_q_all = set()  # 有效召回的query的数量
    print('待评测的query数量：', len(qids_to_ranked_candidate_passages))
    for qid in qids_to_ranked_candidate_passages:  # 枚举所有query
        if qid in qids_to_relevant_passageids:  # qid存在
            tmp_recall_q_top1 = set()
            tmp_recall_q_top50 = set()
            tmp_recall_q_all = set()

            ranking.append(0)
            target_pid = qids_to_relevant_passageids[qid]  # 标准的pid list
            # 模型的pid list
            candidate_pid = qids_to_ranked_candidate_passages[qid]
            for i in range(0, 10):  # mmr@10
                if candidate_pid[i] in target_pid:  # 如果在标准答案中
                    ranking_qid_2_idx_dict_top_10[qid] = i
                    MRR += 1.0 / (i + 1)  # 在参考答案中的位置倒数
                    ranking.pop()
                    ranking.append(i + 1)  # 只记录在模型预测的pid中排在最前面的在标准答案中的位置
                    break
            for i, pid in enumerate(candidate_pid):  # 枚举模型pid list
                if pid == 0:
                    break
                if pid in target_pid:  # 如果在标准答案中
                    recall_q_all.add(qid)
                    if i < 50:
                        ranking_qid_2_idx_dict[qid] = i
                        recall_q_top50.add(qid)
                    if i == 0:
                        recall_q_top1.add(qid)
                    break  # 只记录一个pid
            for i, pid in enumerate(candidate_pid):  # 枚举模型pid list
                if pid == 0:
                    break
                if pid in target_pid:  # 如果在标准答案中
                    tmp_recall_q_all.add(qid)
                    if i < 50:
                        tmp_recall_q_top50.add(qid)
                    if i == 0:
                        tmp_recall_q_top1.add(qid)
            macro_recall1 += len(tmp_recall_q_top1) * 1.0 / len(target_pid)
            macro_recall50 += len(tmp_recall_q_top50) * 1.0 / len(target_pid)
            macro_recallall += len(tmp_recall_q_all) * 1.0 / len(target_pid)

            if qid not in ranking_qid_2_idx_dict:
                not_find_qid.append(qid)
    if len(ranking) == 0:
        raise IOError(
            "No matching QIDs found. Are you sure you are scoring the evaluation set?")

    print('在前10找到答案的query数量：', len(ranking_qid_2_idx_dict_top_10))
    print('在前50找到答案的query数量：', len(ranking_qid_2_idx_dict))
    print('用前10找到答案的做分母的mrr：', MRR / len(ranking_qid_2_idx_dict_top_10))

    MRR = MRR / len(qids_to_relevant_passageids)  # 除以query的个数
    macro_recall1 = macro_recall1 / len(qids_to_relevant_passageids)
    macro_recall50 = macro_recall50 / len(qids_to_relevant_passageids)
    macro_recallall = macro_recallall / len(qids_to_relevant_passageids)
    # 在前1召回正确的qid num/所有qid num
    recall_top1 = len(recall_q_top1) * 1.0 / len(qids_to_relevant_passageids)
    recall_top50 = len(recall_q_top50) * 1.0 / len(qids_to_relevant_passageids)
    recall_all = len(recall_q_all) * 1.0 / len(qids_to_relevant_passageids)
    all_scores['MRR@10'] = MRR
    all_scores["recall@1"] = recall_top1
    all_scores["recall@50"] = recall_top50
    all_scores["recall@all"] = recall_all
    all_scores["macro_recall1"] = macro_recall1
    all_scores["macro_recall50"] = macro_recall50
    all_scores["macro_recallall"] = macro_recallall
    all_scores['QueriesRanked'] = len(qids_to_ranked_candidate_passages)
    return all_scores, ranking_qid_2_idx_dict, ranking_qid_2_idx_dict_top_10, not_find_qid


def compute_metric_pur_example(qids_to_relevant_passageids, qids_to_ranked_candidate_passages, print_path):
    qid2dict = {}
    MRR = 0
    recall1 = 0
    recall50 = 0
    recallall = 0
    print('待评测的query数量：', len(qids_to_ranked_candidate_passages))
    for qid in qids_to_ranked_candidate_passages:  # 枚举所有query
        qid2dict[qid] = {
            'mrr@10': None,
            'macro_recall1': None,
            'macro_recall50': None,
            'macro_recallall': None,
            'recall1': None,
            'recall50': None,
            'recallall': None
        }
        tmp_recall_q_top1 = set()
        tmp_recall_q_top50 = set()
        tmp_recall_q_all = set()
        if qid in qids_to_relevant_passageids:  # qid存在
            target_pid = qids_to_relevant_passageids[qid]  # 标准的pid list
            # 模型的pid list
            candidate_pid = qids_to_ranked_candidate_passages[qid]
            for i in range(0, 10):  # mmr@10
                if candidate_pid[i] in target_pid:  # 如果在标准答案中
                    MRR += 1.0 / (i + 1)  # 在参考答案中的位置倒数
                    qid2dict[qid]['mrr@10'] = 1.0 / (i + 1)
                    break
            for i, pid in enumerate(candidate_pid):  # 枚举模型pid list
                if pid == 0:
                    break
                if pid in target_pid:  # 如果在标准答案中
                    tmp_recall_q_all.add(qid)
                    if i < 50:
                        tmp_recall_q_top50.add(qid)
                    if i == 0:
                        tmp_recall_q_top1.add(qid)
            qid2dict[qid]['macro_recall1'] = len(
                tmp_recall_q_top1) * 1.0 / len(target_pid)
            qid2dict[qid]['macro_recall50'] = len(
                tmp_recall_q_top50) * 1.0 / len(target_pid)
            qid2dict[qid]['macro_recallall'] = len(
                tmp_recall_q_all) * 1.0 / len(target_pid)

            qid2dict[qid]['recall1'] = 1 if len(tmp_recall_q_top1) != 0 else 0
            qid2dict[qid]['recall50'] = 1 if len(
                tmp_recall_q_top50) != 0 else 0
            qid2dict[qid]['recallall'] = 1 if len(tmp_recall_q_all) != 0 else 0

            recall1 += qid2dict[qid]['macro_recall1']
            recall50 += qid2dict[qid]['macro_recall50']
            recallall += qid2dict[qid]['macro_recallall']

            qid2dict[qid]['recall1_size'] = len(tmp_recall_q_top1)
            qid2dict[qid]['recall50_size'] = len(tmp_recall_q_top50)
            qid2dict[qid]['recallall_size'] = len(tmp_recall_q_all)
            qid2dict[qid]['ground_true_size'] = len(target_pid)
    MRR = MRR / len(qids_to_relevant_passageids)  # 除以query的个数
    recall1 = recall1 / len(qids_to_relevant_passageids)
    recall50 = recall50 / len(qids_to_relevant_passageids)
    recallall = recallall / len(qids_to_relevant_passageids)
    all_scores = {}
    all_scores['MRR@10'] = MRR
    all_scores["macro_recall1"] = recall1
    all_scores["macro_recall50"] = recall50
    all_scores["macro_recallall"] = recallall
    all_scores['QueriesRanked'] = len(qids_to_ranked_candidate_passages)
    print(all_scores)
    with open(print_path, 'w', encoding='utf-8') as f_out:
        json.dump(qid2dict, f_out, ensure_ascii=False, indent=2)


def read_reference_table(option='dev'):
    # 从odps读取验证集
    from pypai.io import TableReader
    from aistudio_common.utils import env_utils
    if option == 'dev':
        project_table_name = 'seralgo_dev.t_xinzong_Amazon_relevant_eval_intindex_sel'
    elif option == 'test':
        project_table_name = 'seralgo_dev.t_xinzong_Amazon_relevant_test_intindex'
    o = env_utils.get_odps_instance()
    reader = TableReader.from_ODPS_type(o, project_table_name)
    # 返回全量数据
    dev_df = reader.to_pandas()
    dev_df.head()
    match_dict = {}
    for index, row in dev_df.iterrows():
        if row["query_id"] not in match_dict.keys():
            match_dict[row["query_id"]] = []
        if row["esci_label"] == "E":
            match_dict[row["query_id"]].append(row["int_id"])
    return match_dict


def read_reference_jsonl(option='dev'):
    import jsonlines
    if option == 'dev':
        file_path = '/path/attempt-code/downloads/amazon_data_me/small_amazon_product_eval.jsonl'
    else:
        file_path = '/path/attempt-code/downloads/amazon_data_me/small_amazon_product_test.jsonl'

    match_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for item in jsonlines.Reader(f):
            if item['query_id'] not in match_dict:
                match_dict[item['query_id']] = []
            for p in item['positive_passages']:
                match_dict[item['query_id']].append(p['doc_id'])
    return match_dict


def read_model_res(file='/path/attempt-code/output/model_base_913/encode/rank_test.txt'):
    import pandas as pd
    rank_result = pd.read_csv(file, sep="\t", names=["qid", "pid", "score"])
    result_dict = {}
    for index, row in rank_result.iterrows():
        if str(row["qid"]) not in result_dict.keys():
            result_dict[str(row["qid"])] = []
        result_dict[str(row["qid"])].append(row["pid"])
    return result_dict


def main():
    if len(sys.argv) == 3:
        eval_type = sys.argv[1]  # dev or test
        path_to_eval = sys.argv[2]  # 评测文件
        print_path = "no"
    elif len(sys.argv) == 4:
        eval_type = sys.argv[1]  # dev or test
        path_to_eval = sys.argv[2]  # 评测文件
        print_path = sys.argv[3]
    else:
        print('Usage: result_eval.py <reference ranking> <candidate ranking>')
        exit()
    print(eval_type)
    print(path_to_eval)
    print('print_path:', print_path)

    # 检查每个qid中的pids是否有重复的文章
    qids_to_relevant_passageids = read_reference_jsonl(eval_type)
    qids_to_ranked_candidate_passages = read_model_res(path_to_eval)
    allowed, message = quality_checks_qids(
        qids_to_relevant_passageids, qids_to_ranked_candidate_passages)
    if message != '':
        print(message)
    if print_path == 'no':
        metrics, _, _, _ = compute_metrics(
            qids_to_relevant_passageids, qids_to_ranked_candidate_passages)
        print(metrics)
    else:
        compute_metric_pur_example(
            qids_to_relevant_passageids, qids_to_ranked_candidate_passages, print_path)


if __name__ == "__main__":
    main()
