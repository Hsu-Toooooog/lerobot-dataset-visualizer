from typing import List
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import time
import plotly.graph_objects as go

from loaders.data_loader import (
    Episode,
    Frame,
    FrameRGB,
    FrameState,
    LerobotDataset,
    load_dataset,
)

def render_sidebar(dataset: LerobotDataset) -> None:
    """渲染侧边栏控件"""
    st.sidebar.title("控制面板")

    # Episode选择
    st.session_state.selected_episode = st.sidebar.selectbox(
        "选择 Episode",
        list(range(dataset.num_episodes)),
        index=st.session_state.get("selected_episode", 0),
        format_func=lambda i: f"Episode {i}",
    )

    ep_len = dataset.episodes[st.session_state.selected_episode].length

    # 帧滑块
    st.session_state.current_frame = st.sidebar.slider(
        "选择帧",
        min_value=0,
        max_value=ep_len - 1,
        value=st.session_state.get("current_frame", 0)
    )

    # 播放控制
    cols = st.sidebar.columns(3)
    if cols[0].button("播放/暂停", width='stretch'):
        st.session_state.playing = not st.session_state.get("playing", False)
        # 重置计时，避免切换状态时产生时间累积
        st.session_state.last_tick = None
    if cols[1].button("单步前进", width='stretch'):
        st.session_state.current_frame = min(st.session_state.current_frame + 1, ep_len - 1)
        st.session_state.playing = False
        st.session_state.last_tick = None
    if cols[2].button("单步后退", width='stretch'):
        st.session_state.current_frame = max(st.session_state.current_frame - 1, 0)
        st.session_state.playing = False
        st.session_state.last_tick = None

    # 跨 episode 对比
    st.sidebar.markdown("---")
    st.sidebar.subheader("跨Episode对比")
    options = dataset.joint_names + dataset.gripper_names
    st.session_state.selected_joints_for_compare = st.sidebar.multiselect(
        "选择要对比的关节/夹爪",
        options
    )


def render_dataset_path_controls() -> None:
    """侧边栏顶部：数据集路径输入与加载/清空控制（每个用户会话独立）。"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("数据集路径")
    current = st.session_state.get("dataset_path", "")
    new_path = st.sidebar.text_input(
        "数据目录或单个 parquet 文件",
        value=current,
        help="支持目录或单个 .parquet 文件路径。多个用户可分别设置各自路径。",
        key="dataset_path_input",
    )
    cols = st.sidebar.columns(2)
    if cols[0].button("加载数据", use_container_width=True):
        st.session_state["dataset_path"] = new_path.strip()
        st.session_state.last_tick = None
        st.rerun()
    if cols[1].button("清空路径", use_container_width=True):
        st.session_state.pop("dataset_path", None)
        st.session_state.last_tick = None
        st.rerun()


def render_main_panel(dataset: LerobotDataset) -> None:
    """渲染主显示区域"""
    st.title("LeRobot 数据集可视化")

    selected_ep_idx = st.session_state.selected_episode
    current_frame_idx = st.session_state.current_frame
    episode = dataset.episodes[selected_ep_idx]
    frame = episode.frames[current_frame_idx]

    st.header(f"Episode {selected_ep_idx} / Frame {current_frame_idx}")
    st.subheader("相机视图")
    cols = st.columns(len(dataset.camera_names))
    for i, cam_name in enumerate(dataset.camera_names):
        with cols[i]:
            img = frame.rgb.images.get(cam_name)
            if img is not None:
                st.image(img, caption=cam_name, width='stretch')
            else:
                st.info(f"{cam_name}: 无图像")

    # 2. 当前帧 joints + gripper
    st.subheader("当前帧数据")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown("**关节角 (radians)**")
        joint_data = {"关节": dataset.joint_names, "角度": frame.state.positions}
        st.dataframe(pd.DataFrame(joint_data), width='stretch', height=285)
    with col_b:
        st.metric("夹爪开合度", f"{frame.state.gripper:.3f}")

    # 3. 本 episode 曲线
    st.subheader("本Episode关节/夹爪曲线")
    selected_ep_features = st.multiselect(
        "选择要显示的曲线",
        dataset.joint_names + dataset.gripper_names,
        key="selected_joints_for_episode",
        default=[],
        help="为空则不渲染，选择后仅渲染所选曲线，减少绘图开销。",
    )
    render_episode_curves(dataset, selected_ep_idx, current_frame_idx, selected_ep_features)

    # 4. 跨 episode 对比
    render_comparison_charts(dataset)


def render_episode_curves(dataset: LerobotDataset, ep_idx: int, frame_idx: int, selected_features: List[str]) -> None:
    """仅在选择了要显示的曲线后渲染；对每个 episode+选择 组合只构建一次图，播放时仅更新竖线。"""
    if not selected_features:
        st.info("未选择曲线:为节省计算, 不渲染本Episode曲线。")
        return

    episode = dataset.episodes[ep_idx]
    # 规范化选择顺序，避免相同集合不同顺序造成重复缓存
    features = list(sorted(selected_features))
    sel_key = "|".join(features)

    fig_key = f"ep_fig_{ep_idx}_{sel_key}"
    vline_key = f"ep_vlines_{ep_idx}_{sel_key}"

    # 构建静态曲线（仅一次）
    if fig_key not in st.session_state:
        fig, axes = plt.subplots(len(features), 1, figsize=(10, 2 * len(features)), sharex=True)
        # 当只绘制一条曲线时，axes 不是数组，这里统一成列表处理
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]
        fig.tight_layout(pad=3.0)

        # 预先抽取数据以减少重复遍历
        pos_mat = np.stack([f.state.positions for f in episode.frames], axis=0)
        grip = np.asarray([f.state.gripper for f in episode.frames], dtype=np.float32)

        for ax, name in zip(axes, features):
            if name in dataset.gripper_names:
                ax.plot(grip, label=name, color="orange")
                ax.set_title("Gripper", loc="left")
            else:
                j = dataset.joint_names.index(name)
                ax.plot(pos_mat[:, j], label=name)
                ax.set_title(name, loc="left")
                ax.set_ylabel("position (rad)")
            ax.set_xlabel("frame")
            ax.grid(True, alpha=0.3)

        st.session_state[fig_key] = fig
        st.session_state[vline_key] = []

    fig: plt.Figure = st.session_state[fig_key]
    axes = fig.axes  # type: ignore[attr-defined]

    # 移除旧的竖线，避免叠加
    old_vlines = st.session_state.get(vline_key, []) or []
    for ln in old_vlines:
        try:
            ln.remove()
        except Exception:
            pass
    st.session_state[vline_key] = []

    # 添加新的竖线到所有子图
    new_vlines = []
    for ax in axes:
        ln = ax.axvline(frame_idx, color="r", linestyle="--", lw=1)
        new_vlines.append(ln)
    st.session_state[vline_key] = new_vlines

    st.pyplot(fig)


def render_comparison_charts(dataset: LerobotDataset) -> None:
    selected_features = st.session_state.get("selected_joints_for_compare", [])
    if not selected_features:
        with st.expander("跨Episode对比图",expanded=True):
            st.info("请在左侧控制面板中选择要对比的关节或夹爪。")
        return

    st.subheader("跨Episode数据对比")

    show_labels = st.checkbox(
        "显示末端标签",
        value=st.session_state.get("compare_show_labels", True),
        key="compare_show_labels",
        help="在每条曲线末端显示 Ep 编号，便于识别。过多时可能重叠。",
    )

    for feature_name in selected_features:
        fig = build_plotly_comparison_figure(dataset, feature_name, show_labels=show_labels)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def build_plotly_comparison_figure(
    dataset: LerobotDataset,
    feature_name: str,
    show_labels: bool = True,
):
    is_gripper = feature_name in set(dataset.gripper_names)
    if not is_gripper:
        joint_idx = dataset.joint_names.index(feature_name)

    fig = go.Figure()
    for i, ep in enumerate(dataset.episodes):
        if is_gripper:
            data = np.asarray([f.state.gripper for f in ep.frames], dtype=np.float32)
        else:
            data = np.stack([f.state.positions for f in ep.frames], axis=0)[:, joint_idx]

        fig.add_trace(
            go.Scatter(
                x=np.arange(len(data)),
                y=data,
                mode="lines",
                name=f"Ep {i}",
                line=dict(width=1.8),
                opacity=0.85,
                hovertemplate="Ep %{fullData.name|s}<br>frame=%{x}<br>value=%{y:.3f}<extra></extra>",
            )
        )

    # 末端标注：在每条曲线末端添加 Ep 标签，便于肉眼识别
    if show_labels and len(dataset.episodes) <= 60:
        # 计算一个小的 y 偏移，减少重叠（简单轮换）
        offsets = [-0.02, 0.0, 0.02, 0.04, -0.04]
        for i, ep in enumerate(dataset.episodes):
            if is_gripper:
                data = np.asarray([f.state.gripper for f in ep.frames], dtype=np.float32)
            else:
                data = np.stack([f.state.positions for f in ep.frames], axis=0)[:, joint_idx]
            if len(data) == 0:
                continue
            x_last = len(data) - 1
            y_last = float(data[-1])
            # 使用数据标准差作为偏移尺度，减少重叠
            scale = float(np.nanstd(data) + 1e-6)
            fig.add_annotation(
                x=x_last,
                y=y_last + offsets[i % len(offsets)] * scale,
                xref="x",
                yref="y",
                text=f"Ep {i}",
                showarrow=False,
                font=dict(size=10),
                bgcolor="rgba(255,255,255,0.6)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1,
                xanchor="left",
                yanchor="middle",
            )

    fig.update_layout(
        height=320,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title_text=("Gripper" if is_gripper else feature_name),
    )
    fig.update_xaxes(title_text="frame")
    if not is_gripper:
        fig.update_yaxes(title_text="position (rad)")

    return fig


def init_conf() -> None:
    st.set_page_config(layout="wide")
    default_conf = {"current_frame": 0, "selected_episode": 0, "playing": False, "last_tick": None, "play_fps": 2}
    for key, value in default_conf.items():
        if key not in st.session_state:
            st.session_state.setdefault(key, value)

def auto_play(ep_len):
    if not st.session_state.get("playing", False):
        return

    # 目标帧间隔
    fps = int(st.session_state.get("play_fps", 2))
    interval = 1.0 / fps

    now = time.perf_counter()
    last = st.session_state.get("last_tick")

    # 首次进入播放：记录时间并等待一个间隔后再刷新，给前一次渲染留出时间
    if last is None:
        st.session_state.last_tick = now
        time.sleep(interval)
        st.rerun()
        return

    dt = now - last
    if dt < interval:
        # 还没到下一帧时间，睡到目标间隔再刷新
        time.sleep(max(0.0, interval - dt))
        st.rerun()
        return

    # 计算需要前进的帧数（当渲染慢时一次前进多帧，避免变慢）
    steps = max(1, int(dt / interval))
    st.session_state.current_frame = (st.session_state.current_frame + steps) % ep_len
    # 以 last_tick 为基准累加，降低时间漂移
    st.session_state.last_tick = last + steps * interval
    st.rerun()

def main() -> None:
    init_conf()
    render_dataset_path_controls()

    data_path = st.session_state.get("dataset_path")
    if not data_path:
        st.info("未设置数据集路径。请在左侧输入路径并点击“加载数据”。")
        return

    try:
        dataset = load_dataset(path=data_path)
    except Exception as e:
        st.error(f"加载数据集时出错: {e}")
        return

    render_sidebar(dataset)
    render_main_panel(dataset)

    ep_len = dataset.episodes[st.session_state.selected_episode].length
    auto_play(ep_len)


if __name__ == "__main__":
    main()
