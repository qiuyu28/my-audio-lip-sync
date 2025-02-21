bl_info = {
    "name": "Audio Lip Sync Pro",
    "author": "Your Name & ChatGPT",
    "version": (1, 3, 0),
    "blender": (4, 3, 0),
    "location": "View3D > UI > Animation",
    "description": "Advanced audio-driven facial animation system with emotion control and physics",
    "warning": "Requires external dependencies (torch, librosa, montreal-forced-aligner)",
    "wiki_url": "",
    "tracker_url": "",
    "support": 'COMMUNITY',
    "category": "Animation"
}

import bpy
import numpy as np
import threading
import traceback
from bpy.app.translations import pgettext
from mathutils import Euler
from datetime import datetime

# ========================
# 依赖检查
# ========================
try:
    import torch
    import librosa
    from montreal_forced_aligner import align
    DEPS_INSTALLED = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    DEPS_INSTALLED = False

# ========================
# 多语言支持
# ========================
translation_dict = {
    "en_US": {
        ("*", "Audio File"): "Audio File",
        ("*", "Mouth Bone"): "Mouth Bone",
        ("*", "Eye Bone"): "Eye Bone",
        ("*", "Frame Rate"): "Frame Rate",
        ("*", "Emotion"): "Emotion",
        ("*", "Generate Animation"): "Generate Animation",
        ("*", "Processing..."): "Processing...",
        ("*", "Select Model File"): "Select Model File",
        ("*", "GPU Acceleration"): "GPU Acceleration",
        ("*", "Lip Sync Settings"): "Lip Sync Settings",
        ("*", "Advanced Settings"): "Advanced Settings",
        ("*", "Phonetic Text"): "Phonetic Text",
        ("*", "Align Audio"): "Align Audio",
        ("*", "Waveform Preview"): "Waveform Preview",
        ("*", "Add Physics"): "Add Physics",
    },
    "zh_CN": {
        ("*", "Audio File"): "音频文件",
        ("*", "Mouth Bone"): "嘴部骨骼",
        ("*", "Eye Bone"): "眼部骨骼",
        ("*", "Frame Rate"): "帧率",
        ("*", "Emotion"): "情绪",
        ("*", "Generate Animation"): "生成动画",
        ("*", "Processing..."): "处理中...",
        ("*", "Select Model File"): "选择模型文件",
        ("*", "GPU Acceleration"): "GPU加速",
        ("*", "Lip Sync Settings"): "口型同步设置",
        ("*", "Advanced Settings"): "高级设置",
        ("*", "Phonetic Text"): "音素文本",
        ("*", "Align Audio"): "对齐音频",
        ("*", "Waveform Preview"): "波形预览",
        ("*", "Add Physics"): "添加物理效果",
    }
}

# ========================
# 属性组
# ========================
class LipSyncProperties(bpy.types.PropertyGroup):
    audio_path: bpy.props.StringProperty(
        name="Audio File",
        subtype='FILE_PATH',
        description="Select audio file for lip sync"
    )
    model_path: bpy.props.StringProperty(
        name="Model File",
        subtype='FILE_PATH',
        description="Select trained PyTorch model"
    )
    mouth_bone: bpy.props.StringProperty(
        name="Mouth Bone",
        description="Select mouth control bone"
    )
    eye_bone: bpy.props.StringProperty(
        name="Eye Bone",
        description="Select eye blink control bone"
    )
    frame_rate: bpy.props.EnumProperty(
        name="Frame Rate",
        items=[
            ('24', "24 FPS", ""),
            ('30', "30 FPS", ""),
            ('60', "60 FPS", ""),
        ],
        default='30'
    )
    emotion: bpy.props.EnumProperty(
        name="Emotion",
        items=[
            ('neutral', "Neutral", ""),
            ('happy', "Happy", ""),
            ('angry', "Angry", ""),
            ('sad', "Sad", ""),
            ('crying', "Crying", ""),
            ('shouting', "Shouting", ""),
            ('fear', "Fear", ""),
            ('helpless', "Helpless", ""),
        ],
        default='neutral'
    )
    use_gpu: bpy.props.BoolProperty(
        name="GPU Acceleration",
        default=True,
        description="Enable GPU acceleration"
    )
    progress: bpy.props.FloatProperty(
        name="Progress",
        min=0.0,
        max=1.0,
        default=0.0
    )
    lip_sync_range: bpy.props.FloatVectorProperty(
        name="Lip Sync Range",
        default=(0.0, 1.0),
        min=0.0,
        max=1.0,
        size=2,
        description="Range of audio to process"
    )
    phonetic_text: bpy.props.StringProperty(
        name="Phonetic Text",
        description="Input phonetic text for alignment"
    )
    show_waveform: bpy.props.BoolProperty(
        name="Show Waveform",
        default=False,
        description="Display audio waveform preview"
    )

# ========================
# 音频处理模块
# ========================
class AudioProcessor:
    def __init__(self, filepath, time_range=(0.0, 1.0)):
        self.audio, self.sr = librosa.load(filepath, sr=None)
        start = int(time_range[0] * len(self.audio))
        end = int(time_range[1] * len(self.audio))
        self.audio = self.audio[start:end]
        self.audio_length = len(self.audio) / self.sr
        
    def extract_mfcc(self):
        return librosa.feature.mfcc(y=self.audio, sr=self.sr, n_mfcc=13)

    def get_waveform(self):
        times = np.arange(len(self.audio)) / self.sr
        return times, self.audio

# ========================
# 音素对齐模块
# ========================
class PhoneticAligner:
    def __init__(self):
        self.aligner = align.Aligner()
    
    def align(self, audio_path, text):
        return self.aligner.align(audio_path, text)

# ========================
# 模型推理模块
# ========================
class VisemePredictor:
    emotion_map = {
        'neutral': 0,
        'happy': 1,
        'angry': 2,
        'sad': 3,
        'crying': 4,
        'shouting': 5,
        'fear': 6,
        'helpless': 7
    }

    def __init__(self, model_path, use_gpu=True):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(model_path).to(self.device)
        self.model.eval()

    def predict(self, features, emotion):
        emotion_code = self.emotion_map[emotion]
        inputs = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        emotion_tensor = torch.LongTensor([emotion_code]).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(inputs, emotion_tensor)
        
        return outputs.squeeze().cpu().numpy()

# ========================
# 动画生成模块
# ========================
class AnimationGenerator:
    def __init__(self, context):
        self.scene = context.scene
        self.props = context.scene.lip_sync_props
        self.mouth_bone = bpy.data.objects[self.props.mouth_bone]
        self.eyebone = bpy.data.objects[self.props.eyebone]
        self.fps = int(self.props.frame_rate)
        
    def _insert_keyframe(self, bone, rotation):
        bone.rotation_euler = Euler(rotation)
        bone.keyframe_insert(data_path="rotation_euler", frame=self.scene.frame_current)

    def generate_mouth_animation(self, viseme_data):
        frame_time = 1 / self.fps
        total_frames = int(self.props.audio_length * self.fps)
        
        for frame in range(total_frames):
            self.scene.frame_set(frame)
            time = frame * frame_time
            self._insert_keyframe(self.mouth_bone, viseme_data[frame])

    def generate_blink_animation(self):
        rng = np.random.default_rng()
        blink_times = rng.poisson(lam=5, size=10)
        
        for t in blink_times:
            frame = int(t * self.fps)
            for i in range(3):
                self.scene.frame_set(frame + i)
                rotation = (0, 0, 0.5 if i == 1 else 0)
                self._insert_keyframe(self.eyebone, rotation)

    def add_physics(self):
        bpy.ops.object.select_all(action='DESELECT')
        self.mouth_bone.select_set(True)
        bpy.context.view_layer.objects.active = self.mouth_bone
        bpy.ops.rigidbody.object_add()
        bpy.context.object.rigid_body.collision_shape = 'MESH'
        bpy.context.object.rigid_body.mass = 0.5
        bpy.context.object.rigid_body.friction = 0.5
        bpy.context.object.rigid_body.restitution = 0.1

# ========================
# 主操作符
# ========================
class LIPSYNC_OT_Generate(bpy.types.Operator):
    bl_idname = "lipsync.generate"
    bl_label = "Generate Lip Sync Animation"
    bl_options = {'REGISTER', 'UNDO'}

    _timer = None
    _thread = None
    _cancel = False

    def modal(self, context, event):
        if event.type == 'TIMER':
            if self._thread.is_alive():
                context.area.tag_redraw()
                return {'PASS_THROUGH'}
            
            self.finish(context)
            return {'FINISHED'}
        return {'PASS_THROUGH'}

    def execute(self, context):
        if not DEPS_INSTALLED:
            self.report({'ERROR'}, "Missing dependencies. Please install torch, librosa, and montreal-forced-aligner.")
            return {'CANCELLED'}

        props = context.scene.lip_sync_props
        
        if not props.audio_path:
            self.report({'ERROR'}, "请先选择音频文件")
            return {'CANCELLED'}
            
        if not props.mouth_bone:
            self.report({'ERROR'}, "请选择嘴部骨骼")
            return {'CANCELLED'}

        self._thread = threading.Thread(target=self._process_data, args=(context,))
        self._thread.start()

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _process_data(self, context):
        try:
            props = context.scene.lip_sync_props
            
            audio_processor = AudioProcessor(props.audio_path, props.lip_sync_range)
            mfcc_features = audio_processor.extract_mfcc()
            
            if props.phonetic_text:
                aligner = PhoneticAligner()
                alignment = aligner.align(props.audio_path, props.phonetic_text)
            
            predictor = VisemePredictor(props.model_path, props.use_gpu)
            viseme_data = predictor.predict(mfcc_features, props.emotion)
            
            anim_gen = AnimationGenerator(context)
            anim_gen.generate_mouth_animation(viseme_data)
            anim_gen.generate_blink_animation()
            
            if props.add_physics:
                anim_gen.add_physics()
            
        except Exception as e:
            self.report({'ERROR'}, str(e))
            traceback.print_exc()
            self._cancel = True

    def finish(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        context.scene.lip_sync_props.progress = 0.0
        self._thread.join()

# ========================
# UI面板
# ========================
class LIPSYNC_PT_MainPanel(bpy.types.Panel):
    bl_label = "口型同步"
    bl_idname = "LIPSYNC_PT_MainPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Animation"

    def draw(self, context):
        layout = self.layout
        props = context.scene.lip_sync_props
        
        if not DEPS_INSTALLED:
            layout.label(text="Missing dependencies!", icon='ERROR')
            layout.label(text="Please install torch, librosa, and montreal-forced-aligner.")
            return

        box = layout.box()
        box.label(text=pgettext("Lip Sync Settings"))
        box.prop(props, "audio_path", text=pgettext("Audio File"))
        box.prop(props, "model_path", text=pgettext("Select Model File"))
        
        row = box.row()
        row.prop_search(props, "mouth_bone", context.scene, "objects", text=pgettext("Mouth Bone"))
        row.prop_search(props, "eyebone", context.scene, "objects", text=pgettext("Eye Bone"))
        
        box = layout.box()
        box.label(text=pgettext("Advanced Settings"))
        box.prop(props, "frame_rate", text=pgettext("Frame Rate"))
        box.prop(props, "emotion", text=pgettext("Emotion"))
        box.prop(props, "use_gpu", text=pgettext("GPU Acceleration"))
        box.prop(props, "lip_sync_range", text="Audio Range")
        box.prop(props, "phonetic_text", text=pgettext("Phonetic Text"))
        box.prop(props, "show_waveform", text=pgettext("Waveform Preview"))
        
        if props.show_waveform:
            draw_waveform(context, box)
        
        if props.progress > 0:
            box.prop(props, "progress", slider=True, text=pgettext("Processing..."))
        
        row = layout.row()
        row.operator(LIPSYNC_OT_Generate.bl_idname, text=pgettext("Generate Animation"))
        row.operator("lipsync.add_physics", text=pgettext("Add Physics"))

# ========================
# 注册/注销
# ========================
classes = (
    LipSyncProperties,
    LIPSYNC_OT_Generate,
    LIPSYNC_OT_AddPhysics,
    LIPSYNC_PT_MainPanel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.lip_sync_props = bpy.props.PointerProperty(type=LipSyncProperties)
    bpy.app.translations.register(__name__, translation_dict)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.lip_sync_props
    bpy.app.translations.unregister(__name__)

if __name__ == "__main__":
    register()
