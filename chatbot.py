import json
import os
import traceback
import uuid
from copy import deepcopy
from flask import request, Flask
import openai
import requests
from transformers import GPT2TokenizerFast

with open("config.json", "r", encoding='utf-8') as jsonfile:
    config_data = json.load(jsonfile)

session_config = {
    'preset': '你是 ChatBot，一个大型语言模型，由 OpenAI 训练而成。你是一种人工智能程序，可以回答我的问题。如果我有任何问题，会随时告诉你，你会尽力为我解答。',
    'context': ''
}

sessions = {}

# 创建一个服务，把当前这个python文件当做一个服务
server = Flask(__name__)

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


# 测试接口，可以测试本代码是否正常启动
@server.route('/', methods=["PUT"])
def index():
    return f"你好，QQ机器人逻辑处理端已启动<br/>"

# 测试接口，可以用来测试与ChatGPT的交互是否正常，用来排查问题
@server.route('/chat', methods=['post'])
def chatapi():
    requestJson = request.get_data()
    if requestJson is None or requestJson == "" or requestJson == {}:
        resu = {'code': 1, 'data':{'msg': '请求内容不能为空'}}
        return json.dumps(resu, ensure_ascii=False)
    data = json.loads(requestJson)
    print(data)
    try:
        msgFromSender = data['msg']
        senderSessionId = data['uuid']
        if senderSessionId is None or senderSessionId == "":
            resu = {'code': -1, 'data':{'msg':'请求错误'}}
            return json.dumps(resu, ensure_ascii=False)
        msg = chat(msgFromSender, senderSessionId)
        msg = msg.strip()
        resu = {'code': 0, 'data': {'msg': msg}}
        return json.dumps(resu, ensure_ascii=False)
    except Exception as error:
        print("接口报错")
        resu = {'code': 1, 'data':{'msg': '请求异常: ' + str(error)}}
        return json.dumps(resu, ensure_ascii=False)


# 与ChatGPT交互的方法
def chat(msg, sessionid):
    try:
        if msg.strip() == '':
            return '您好，我是人工智能助手，如果您有任何问题，请随时告诉我，我将尽力回答。\n如果您需要重置我们的会话，请回复`重置会话`'
        # 获得对话session
        session = get_chat_session(sessionid)
        if '重置会话' == msg.strip():
            session['context'] = ''
            return "会话已重置"
        if '重置人格' == msg.strip():
            session['context'] = ''
            session['preset'] = session_config['preset']
            return '人格已重置'
        if '指令说明' == msg.strip():
            return "指令如下：\n1.[重置会话] 请发送 重置会话\n2.[设置人格] 请发送 设置人格+人格描述\n3.[重置人格] 请发送 重置人格\n4.[指令说明] 请发送 " \
                   "指令说明\n注意：\n重置会话不会清空人格,重置人格会重置会话!\n设置人格后人格将一直存在，除非重置人格或重启逻辑端!"
        if msg.strip().startswith('设置人格'):
            session['preset'] = msg.strip().replace('设置人格', '')
            session['context'] = ''
            return '人格设置成功'
        # 处理上下文逻辑
        token_limit = 4096 - config_data['chatgpt']['max_tokens'] - len(tokenizer.encode(session['preset'])) - 3
        session['context'] = session['context'] + "\n\nQ:" + msg + "\nA:"
        ids = tokenizer.encode(session['context'])
        tokens = tokenizer.decode(ids[-token_limit:])
        # 计算可发送的字符数量
        char_limit = len(''.join(tokens))
        session['context'] = session['context'][-char_limit:]
        # 从最早的提问开始截取
        pos = session['context'].find('Q:')
        session['context'] = session['context'][pos:]
        # 设置预设
        msg = session['preset'] + '\n\n' + session['context']
        # 与ChatGPT交互获得对话内容
        message = chat_with_gpt(msg)
        print("会话ID: " + str(sessionid))
        print("ChatGPT返回内容: ")
        print(message)
        return message
    except Exception as error:
        traceback.print_exc()
        return str('异常: ' + str(error))


# 获取对话session
def get_chat_session(sessionid):
    if sessionid not in sessions:
        config = deepcopy(session_config)
        config['id'] = sessionid
        sessions[sessionid] = config
    return sessions[sessionid]


def chat_with_gpt(prompt):
    try:
        if not config_data['openai']['api_key']:
            return "请设置Api Key"
        else:
            openai.api_key = config_data['openai']['api_key']
        resp = openai.Completion.create(**config_data['chatgpt'], prompt=prompt)
        resp = resp['choices'][0]['text']
    except openai.OpenAIError as e:
        print('openai 接口报错: ' + str(e))
        resp = str(e)
    return resp


if __name__ == '__main__':
    server.run(port=5555, host='0.0.0.0', use_reloader=False)
