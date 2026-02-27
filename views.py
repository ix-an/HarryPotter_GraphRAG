from django.http import JsonResponse, StreamingHttpResponse
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.utils import timezone
import base64
import os
import time
from utils.QwenLLM import QwenLLM
from .models import Conversation,Theme
from utils.RAGSystem import RAGSystem
from GraphRAG.grag_utils.grag_system import GraphRAGService


# 初始化 QwenLLM类
qwen = QwenLLM()
# 初始化 RAGSystem类
rag = RAGSystem()

# ===================== 工具函数 =====================
# 图片编码函数：将本地文件转为base64编码的字符串
def encode_image(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


# ===================== 核心业务接口 =====================
# 助手聊天接口：处理用户对话请求
def assistant(request):
    # 接受客户端的 POST请求，获取请求数据
    query = request.POST.get('query', '用户未输入任何内容')
    user_id = int(request.POST.get('user_id', 1))

    # --------------------------------------------------
    # 对话主题处理：无主题时自动生成并创建主题，有主题时复用
    # --------------------------------------------------
    # 接收前端传入的 对话主题 id
    theme_id = int(request.POST.get('theme_id', 0))

    if theme_id == 0:
        # 存储对话信息到 Theme表,并获取对话 id
        theme_prompt = f'''
        请严格按照以下要求处理用户提问，生成对话主题：
        1. 核心要求：仅提取用户提问的核心意图，生成20字以内的简短主题；
        2. 输出规则：只返回主题文本，无任何解释、标点、多余内容；
        3. 示例：
           - 用户提问：“苹果的热量是多少？适合减肥吃吗？” → 输出：减脂期水果
           - 用户提问：“早餐吃燕麦和鸡蛋好不好” → 输出：早餐食谱
           - 用户提问：“帮我推荐减脂期的晚餐” → 输出：减脂期晚餐推荐
           
        用户提问：{query}
        '''
        theme_name = qwen.inference(
            messages=[{'role': 'system','content': theme_prompt}],
            model="qwen-flash",
            max_tokens=30,
        )
        theme = Theme.objects.create(
            user_id=user_id,
            theme_name=theme_name,
            create_time=timezone.now(),
            update_time=timezone.now(),
        )
        theme_id = theme.id

    # 初始化模型系统提示：定义助手角色
    msg = [{'role': 'system', 'content': '你叫柠柠，是一个小说迷，特别喜欢《哈利波特》，可以和用户讨论小说相关内容。'}]


    # 历史对话加载：从数据库获取当前主题下的所有有效对话记录
    full_history = Conversation.objects.filter(
        user_id=user_id,
        theme_id=theme_id,
        is_deleted=0,
    ).order_by('id')  # 从旧到新

    # 转为 Python 列表，方便切片
    full_list = list(full_history)

    # --------------------------------------------------
    # 记忆分层处理：短期记忆（最近10条）+ 长期记忆（更早记录摘要）
    # --------------------------------------------------
    long_term = []
    long_term_summary = ""
    if len(full_list) <= 10:
        # 全部作为短期记忆
        short_term = full_list
    else:
        short_term = full_list[-10:]  # 最近10条 → 短期
        long_term = full_list[:-10]  # 更早的 → 需要摘要

    # 如果有长期记忆，调用模型生成摘要
    if long_term:
        # 拼接长期记忆的对话文本（按角色+内容格式）
        history_text = "\n".join([
            f"[{item.role.upper()}]: {item.content}"
            for item in long_term
        ])

        # 摘要提示词
        summary_prompt = (f'''
        你是一个对话摘要助手。请将以下用户与AI的历史对话内容，压缩成一段不超过100字的简洁摘要，保留核心信息和用户意图。\n\n
        历史对话：
        {history_text}
        ''')

        try:
            # 调用模型生成长期记忆摘要
            summary_resp = qwen.inference(
                messages=[{"role": "user", "content": summary_prompt}],
                model="qwen-flash",
            )
            long_term_summary = summary_resp.strip()
        except Exception as e:
            print(f"摘要生成失败: {e}")
            long_term_summary = ""  # 失败时留空，不影响主流程

    # --------------------------------------------------
    # 记忆组装：将长期记忆摘要+短期记忆完整记录加入对话上下文
    # --------------------------------------------------
    # 长期记忆
    if long_term_summary:
        msg.append({
            'role': 'system',
            'content': f'【历史对话摘要】{long_term_summary}'
        })
    # 短期记忆
    for item in short_term:
        msg.append({'role': item.role, 'content': item.content})

    # --------------------------------------------------
    # GraphRAG检索：优先使用知识图谱检索
    # --------------------------------------------------
    graphrag_chunks = None
    try:
        print("\n🤖 开始 GraphRAG 检索...")
        graphrag_chunks = GraphRAGService.retrieval_chunks(query)
        if graphrag_chunks and graphrag_chunks.get("documents") and graphrag_chunks["documents"]:
            grag_text = "\n".join(graphrag_chunks["documents"][:10])  # 最多10条
            print("✅ GraphRAG 检索成功，添加到上下文")
            msg.append({
                'role': 'system',
                'content': f'''
                以下是与用户问题相关的知识图谱参考资料，仅供你回答时参考。
                如果无关请忽略。
                【知识图谱参考资料】
                {grag_text}
                '''
            })
        else:
            print("⚠️  GraphRAG 未检索到相关信息，回退到普通 RAG")
    except Exception as e:
        print("GraphRAG 检索失败：", e)
    
    # --------------------------------------------------
    # 普通RAG检索：当GraphRAG无结果时使用
    # --------------------------------------------------
    if not graphrag_chunks or not graphrag_chunks.get("documents") or not graphrag_chunks["documents"]:
        try:
            print("\n📚 开始普通 RAG 检索...")
            chunks = rag.retrieval_chunks(query)
            if chunks and chunks.get("documents"):
                rag_text = "\n".join(chunks["documents"][:10])  # 最多10条
                print("✅ RAG 检索成功，添加到上下文")
                msg.append({
                    'role': 'system',
                    'content': f'''
                    以下是与用户问题相关的参考资料，仅供你回答时参考。
                    如果无关请忽略。
                    【参考资料】
                    {rag_text}
                    '''
                })
            else:
                print("⚠️  RAG 未检索到相关信息")
        except Exception as e:
            print("RAG 检索失败：", e)

    # 模型配置与图片处理
    model = "qwen-plus"    # 默认模型
    web_flag = request.POST.get('web', '0')    # 联网搜索
    think_flag = request.POST.get('deepthink', '0')    # 深度思考
    image_url = request.POST.get('file_url',  '')    # 接收文件路径

    # --------------------------------------------------
    # 加入用户当前的最新提问 + 图片处理逻辑
    # --------------------------------------------------
    if image_url:    # 如果有图片上传
        # 更换为有视觉处理能力的模型
        model = "qwen3-vl-plus"
        # 获取文件格式后缀
        file_extension = os.path.splitext(image_url)[1]
        # 将图片转为base64编码
        base64_image = encode_image(os.path.join(settings.BASE_DIR, image_url[1:]))
        msg.append(
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f"data:image/{file_extension};base64,{base64_image}",
                        },
                    },
                    {'type': 'text', 'text': query}
                ],
            },
        )
    else:    # 如果没有图片上传
        msg.append({'role': 'user', 'content': query})

    # 保存用户的对话内容到 Conversation表 -> 查询完历史后，调用模型之前保存
    Conversation.objects.create(
        theme_id=theme_id,
        user_id=user_id,
        role='user',
        content=query,
        create_time=timezone.now(),
        update_time=timezone.now(),
        image_url=image_url,
    )

    # 调用QwenLLM类的inference方法，获取模型回复
    answer = qwen.inference(
        messages=msg,
        model=model,
        stream=True,  # 是否流式返回
        enable_search=web_flag == '1',
        enable_thinking=think_flag == '1',
    )

    # 流式推理函数
    def event_stream():
        content = ""  # 用于更新数据库
        try:
            # 响应对话主题id到客户端
            # SSE 协议要求：事件数据块必须以 "data: "开头，之间必须用 "\n\n" 分隔
            yield "data: <theme_id_1>" + str(theme_id) + "<theme_id_1>\n\n"
            # 流式输出
            for chunk in answer:
                if chunk.choices:
                    delta = chunk.choices[0].delta or ""
                    if delta and delta.content:
                        content += delta.content
                        yield f"data: {delta.content.replace("\n", "\\n")}\n\n"
        except Exception as e:
            print(f"流式推理过程发生错误：{e}")
        finally:
            # 把结束标志发送到客户端
            yield "data: [@#--END--#@]\n\n"
            # 保存助手的回复到 Conversation表
            Conversation.objects.create(
                theme_id=theme_id,
                user_id=user_id,
                role='assistant',
                content=content,
                create_time=timezone.now(),
                update_time=timezone.now(),
            )

    response = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
    # 设置缓存控制：只要有yield产生的数据，就立即响应到客户端，不要缓存
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"  # 禁止 Nginx 缓冲
    return response
    # data = {
    #     "status": "success",
    #     "message": query,
    #     "answer": answer,
    #     "searched": web_flag == '1',
    #     "thought": think_flag == '1',
    #     "theme_id": theme_id,
    # }
    # return JsonResponse(data)


# 文件上传接口：接收前端上传的图片文件，保存并返回文件路径
def uploadfile(request):
    # 接受客户端提交的文件
    file = request.FILES.get('file_1')
    # 创建 FileSystemStorage 对象
    fs = FileSystemStorage(
        location=str(settings.BASE_DIR / "static" / "uploads"),  # 文件保存的物理路径
        base_url=settings.STATIC_URL + "uploads/",               # 文件访问的URL前缀
    )

    # 生成带时间戳的文件名
    org_name = file.name
    ext = os.path.splitext(org_name)[1]  # 如 ".jpg"、".png"
    timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
    new_name = f"{org_name}_{timestamp}{ext}"
    
    # 保存文件
    filename = fs.save(new_name, file)
    # 获取文件地址
    file_url = request.build_absolute_uri(fs.url(filename))

    data = {
        "status": "success", 
        "file_url": file_url, 
        "message": "文件上传接口"
    }
    return JsonResponse(data)


# 对话主题列表接口：获取指定用户的所有有效对话主题
def history(request):
    # 接收用户 id
    user_id = int(request.POST.get('user_id', 0))
    # print("DEBUG user_id =", user_id)
    # 获取历史对话主题列表：加 - 表示按创建时间降序排序
    themes = Theme.objects.filter(user_id=user_id, is_deleted=0).order_by('-create_time')
    # 初始化历史消息列表
    history_list = []
    for theme in themes:
        history_list.append({
            "theme_id": theme.id,
            "theme_name": theme.theme_name,
            "create_time": theme.create_time.strftime("%Y-%m-%d %H:%M:%S"),
        })
    # 响应到客户端
    data = {
        "status": "success",
        "message": "对话历史记录接口",
        "history_list": history_list,
    }
    return JsonResponse(data)


# 历史对话详情接口：获取指定主题下的所有对话记录
def continue_history(request):
    # 接收前端传入的参数：用户id + 对话主题id
    user_id = int(request.POST.get('user_id', 0))
    theme_id = int(request.POST.get('theme_id', 0))

    # 查询历史对话消息
    info = Conversation.objects.filter(
        theme_id=theme_id,
        user_id=user_id,
        is_deleted=0,  # 0 表示未删除
    ).order_by('id')

    # 格式化聊天记录
    chat_list = []
    for item in info:
        chat_list.append({
            "role": item.role,
            "content": item.content,
            "image_url": (
                request.build_absolute_uri('/')[:-1] + item.image_url
                if item.image_url else ""
            )
        })
    data = {
        "status": "success",
        "message": "获取历史对话接口",
        "chat": chat_list,
    }
    return JsonResponse(data)

# 对话主题删除接口：逻辑删除指定主题及关联的所有对话记录
def del_theme(request):
    # 接收前端传入的参数：用户id + 对话主题id
    user_id = int(request.POST.get('user_id', 0))
    theme_id = int(request.POST.get('theme_id', 0))

    # ------------------------------
    # 物理删除
    # ------------------------------
    # # 删除对话记录
    # Conversation.objects.filter(theme_id=theme_id, user_id=user_id).delete()
    # # 删除对话主题
    # Theme.objects.filter(id=theme_id, user_id=user_id).delete()

    # ------------------------------
    # 逻辑删除：仅标记删除状态，不删除数据库数据
    # ------------------------------
    Conversation.objects.filter(
        theme_id=theme_id,
        user_id=user_id
    ).update(is_deleted=1)

    Conversation.objects.filter(
        theme_id=theme_id,
        user_id=user_id
    ).update(delete_time=timezone.now())

    Theme.objects.filter(
        id=theme_id,
        user_id=user_id
    ).update(is_deleted=1)

    Theme.objects.filter(
        id=theme_id,
        user_id=user_id
    ).update(delete_time=timezone.now())

    data = {
        "status": "success",
        "message": "删除历史对话记录接口",
    }
    return JsonResponse(data)
