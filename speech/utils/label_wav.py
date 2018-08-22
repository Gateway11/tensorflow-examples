import os
from collections import Counter

def prepare_label_wav_list(wav_files, labels_dict):
    labels = []
    new_wav_files = []
    for wav_file in wav_files:
        wav_id = os.path.basename(wav_file).split(".")[0]
        if wav_id in labels_dict:
            labels.append(labels_dict[wav_id])
            new_wav_files.append(wav_file)

    all_words = []
    for label in labels:
        # 字符分解
        all_words += [word for word in label]

    counter = Counter(all_words)
    count_pairs = sorted(counter.items(), key = lambda x: -x[1])

    words, _ = zip(*count_pairs)
    words_size = len(words)
    print("词汇表大小：", words_size)

    lexicon = dict(zip(words, range(len(words))))

    # 将单个file的标签映射为num 返回对应list,最终all file组成嵌套list
    labels_vector = [words2vec(label, lexicon) for label in labels]

    return lexicon, labels_vector

def words2vec(words, lexicon):
    return list(map(lambda word: lexicon.get(word, len(words)), words))


if __name__ == "__main__":
    wav_files = ['/Users/daixiang/deep-learning/data/data_wsj/wav/train/B21/B21_254.wav', 
                 '/Users/daixiang/deep-learning/data/data_wsj/wav/train/B21/B21_268.wav', 
                 '/Users/daixiang/deep-learning/data/data_wsj/wav/train/B21/B21_250.wav', 
                 '/Users/daixiang/deep-learning/data/data_wsj/wav/train/B21/B21_251.wav', 
                 '/Users/daixiang/deep-learning/data/data_wsj/wav/train/B21/B21_263.wav', 
                 '/Users/daixiang/deep-learning/data/data_wsj/wav/train/B21/B21_262.wav', 
                 '/Users/daixiang/deep-learning/data/data_wsj/wav/train/B21/B21_261.wav', 
                 '/Users/daixiang/deep-learning/data/data_wsj/wav/train/B21/B21_265.wav', 
                 '/Users/daixiang/deep-learning/data/data_wsj/wav/train/B21/B21_259.wav', 
                 '/Users/daixiang/deep-learning/data/data_wsj/wav/train/B21/B21_258.wav']

    labels_dict = {'B21_254': '新 聘 国务院 参事 王 一平 新 聘 中央 文史 研究 馆 馆员 孙 机 程 毅 中 也 应邀 参加 了 招待会', 
                   'B21_268': '碰上 爱玩 且 能 玩物 成 痴 玩物 成 癖 的 人 总能 令人 愉快 地 会心一笑', 
                   'B22_250': '嫌疑犯 眼看 脱逃 无望 窜 上 屋顶 纵身 跳下 意欲 自杀 结果 摔成 骨折 被 生擒', 
                   'B22_251': '且 夫 孥 孥 阿文 确 尚无 偷 文 如 欧阳 公 之 恶 德 而 文章 亦 较为 能 做做 者 也', 
                   'B21_263': '中国 要求 在 别国 遗弃 化学武器 的 国家 应 按 公约 规定 尽快 彻底 销毁 所有 遗弃 化学武器', 
                   'B21_262': '电话 是 市 公安局 派 往 黑龙江省 安达市 的 追捕 小组 打来 的 侦查员 报告 持枪 在逃犯 已 被 生擒', 
                   'B21_261': '大 院里 也是 两派 在 骂 夜晚 也 在 斗 走资派 一天到晚 心惊肉跳 随时 准备 着 挨斗', 
                   'B21_265': '参 战军 虽然 没有 参加 对 德 战争 但 有 十万 华工 被 派 赴 欧洲 战场 为 协约国 军队 运输 弹药 给养 和 修筑 工事', 
                   'B21_259': '千 百年 来 逐 水草 而 居 靠 天 养 畜 的 藏族 牧民 开始 在 围栏 内 划 片 轮 牧 并 利用 农机具 种草', 
                   'B21_258': '要 理顺 产权 关系 除了 要 理顺 政企 关系 外 首先 要 理顺 企业 资产 所有者 与 经营者 的 关系'}

    lexicon, labels_vector = prepare_label_wav_list(wav_files, labels_dict)
    print(wav_files[0])
    print(labels_vector[0])
    print(list(lexicon.keys())[17])
    #print(words[17]) # 新
    #/Users/daixiang/deep-learning/data/data_wsj/wav/train/B21/B21_254.wav 
    #[17, 0, 18, 0, 2, 60, 19, 0, 4, 20, 0, 61, 0, 8, 62, 0, 17, 0, 18, 0, 9, 63, 0, 64, 65, 0, 66, 67, 0, 21, 0, 21, 22, 0, 68, 0, 23, 0, 69, 0, 70, 0, 9, 0, 10, 0, 24, 71, 0, 4, 25, 0, 26, 0, 72, 73, 27]

    label_vector = words2vec(list(labels_dict.values())[0], lexicon)
    print(label_vector)
