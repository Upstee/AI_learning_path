#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
创建VLA基本概念详解.ipynb文件
将VLA基本概念详解.md转换为.ipynb格式，并补充术语表和可视化代码
"""

import json
import os

# 创建notebook的基本结构
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Cell 1: 标题和文档说明
cell1 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# VLA基本概念详解\n",
        "\n",
        "## 📋 文档说明\n",
        "\n",
        "本文档是VLA（Vision-Language-Action）基本概念的详细理论讲解。通过本文档，你将能够：\n",
        "\n",
        "1. **深入理解VLA的定义和核心概念**：从多模态学习到端到端学习，全面掌握VLA的基本概念\n",
        "2. **掌握VLA的数学表示**：理解VLA模型的数学框架和损失函数\n",
        "3. **理解VLA的三个核心模块**：Vision、Language、Action模块的工作原理\n",
        "4. **了解VLA与传统模型的对比**：理解VLA的优势和劣势\n",
        "5. **掌握VLA的关键技术**：多模态融合、预训练、强化学习、推理与规划等\n",
        "\n",
        "**学习方式**：本文件是Jupyter Notebook格式，你可以边看边运行代码，通过可视化图表和数学推导更好地理解VLA的基本概念和原理。\n",
        "\n",
        "---\n",
        "\n",
        "## 📖 论文引用说明\n",
        "\n",
        "本文档引用的论文来自 `VLA/科研论文/` 文件夹，引用格式如下：\n",
        "- `[Survey]` - A Survey on Vision-Language-Action Models\n",
        "- `[openVLA]` - openVLA: An Open-Source Vision-Language-Action Model\n",
        "- `[VLA-R1]` - VLA-R1: Enhancing Reasoning in Vision-Language-Action Models\n",
        "- `[CoA-VLA]` - CoA-VLA: Improving Vision-Language-Action Models via Visual-Text Chain-of-Affordance\n",
        "- `[IntentionVLA]` - IntentionVLA: Generalizable and Efficient Embodied Intention\n",
        "- `[VLASER]` - VLASER: Vision-Language-Action Model\n",
        "- `[Scalable]` - Scalable Vision-Language-Action Model Pretraining\n",
        "- `[Efficient]` - Efficient Vision-Language-Action Models\n",
        "\n",
        "详细引用索引请参考：[论文引用索引.md](./论文引用索引.md)"
    ]
}

notebook["cells"].append(cell1)

# Cell 2: 术语表（简化版，完整版需要更多内容）
cell2 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 📚 术语表（按出现顺序）\n",
        "\n",
        "### 1. Vision-Language-Action (VLA) 模型\n",
        "- **中文名称**：视觉-语言-动作模型\n",
        "- **英文全称**：Vision-Language-Action Model\n",
        "- **定义**：VLA是一种能够同时理解视觉信息、语言指令并生成动作序列的多模态AI模型。它结合了计算机视觉、自然语言处理和机器人控制三个领域的技术，能够根据视觉输入和语言指令，生成相应的动作序列来控制机器人执行任务。VLA的核心特点是端到端学习，即从原始输入（图像和文本）直接到最终输出（动作序列），整个系统作为一个整体进行训练，无需手工设计中间表示。\n",
        "- **核心组成**：VLA模型由三个核心模块组成：1）Vision（视觉）模块：负责从图像或视频中提取视觉特征，理解视觉场景、识别物体、理解空间关系等；2）Language（语言）模块：负责理解自然语言指令，提取语义信息，理解任务要求等；3）Action（动作）模块：负责根据视觉和语言信息生成动作序列，控制机器人执行任务。这三个模块通过多模态融合机制连接，实现端到端的学习和推理。\n",
        "- **在VLA中的应用**：VLA模型是本文档的核心主题，是整个VLA学习体系的基础。\n",
        "- **相关概念**：多模态学习、端到端学习、强化学习、具身智能\n",
        "- **直观理解**：想象一个机器人助手，它能够\"看到\"周围的环境（视觉），\"听懂\"你的指令（语言），然后\"做出\"相应的动作（动作）。VLA模型就是让机器人具备这种能力的技术。\n",
        "\n",
        "### 2. 多模态（Multimodal）\n",
        "- **中文名称**：多模态\n",
        "- **英文全称**：Multimodal\n",
        "- **定义**：多模态是指系统能够处理多种类型的数据（如图像、文本、音频、视频等）。在VLA中，主要处理视觉（图像/视频）和语言（文本）两种模态。多模态学习的核心挑战是如何将不同模态的信息融合，使得模型能够理解跨模态的对应关系。\n",
        "- **在VLA中的应用**：在VLA中，多模态融合是核心能力。VLA模型需要同时理解视觉场景和语言指令，然后将两者融合，生成相应的动作。\n",
        "\n",
        "### 3. 端到端学习（End-to-End Learning）\n",
        "- **中文名称**：端到端学习\n",
        "- **英文全称**：End-to-End Learning\n",
        "- **定义**：端到端学习是指从原始输入到最终输出，整个系统可以作为一个整体进行训练，无需手工设计中间表示。在VLA中，端到端学习意味着从图像和文本直接到动作，整个模型作为一个函数进行优化。\n",
        "- **在VLA中的应用**：在VLA中，端到端学习使得模型能够自动学习最适合任务的特征表示。\n",
        "\n",
        "### 4. 具身智能（Embodied AI）\n",
        "- **中文名称**：具身智能\n",
        "- **英文全称**：Embodied AI\n",
        "- **定义**：具身智能是指智能体具有物理身体，能够在真实或虚拟环境中执行动作，通过与环境交互来学习和完成任务。VLA是实现具身智能的重要技术路径。\n",
        "- **在VLA中的应用**：VLA是实现具身智能的重要技术，因为它整合了感知（视觉）、理解（语言）、决策（融合）和执行（动作）四个环节。"
    ]
}

notebook["cells"].append(cell2)

# Cell 3: 概述
cell3 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 📋 概述\n",
        "\n",
        "### 什么是VLA\n",
        "\n",
        "**VLA**（Vision-Language-Action）是一种**多模态端到端学习系统**，能够同时处理视觉信息、理解自然语言指令，并生成相应的动作序列。`[Survey]` `[openVLA]`\n",
        "\n",
        "### 为什么重要\n",
        "\n",
        "VLA对于实现具身智能非常重要，原因包括：\n",
        "\n",
        "1. **整合感知、理解和执行**：VLA将视觉理解、语言理解和动作生成整合在一个模型中\n",
        "2. **端到端学习**：自动学习最优的特征表示和映射关系\n",
        "3. **更强的泛化能力**：在大量数据上训练，能够泛化到新场景\n",
        "4. **实际应用价值**：可以部署在机器人上，实现真正的智能机器人"
    ]
}

notebook["cells"].append(cell3)

# Cell 4: 可视化代码
cell4 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================\n",
        "# 可视化：VLA模型的基本架构\n",
        "# ============================================\n",
        "import matplotlib.pyplot as plt\n",
        "import matplotlib.patches as patches\n",
        "from matplotlib.patches import FancyBboxPatch, FancyArrowPatch\n",
        "\n",
        "# 创建图形\n",
        "fig, ax = plt.subplots(1, 1, figsize=(14, 8))\n",
        "ax.set_xlim(0, 10)\n",
        "ax.set_ylim(0, 6)\n",
        "ax.axis('off')\n",
        "\n",
        "# 定义颜色\n",
        "color_vision = '#4A90E2'  # 蓝色\n",
        "color_language = '#50C878'  # 绿色\n",
        "color_fusion = '#FF6B6B'  # 红色\n",
        "color_action = '#FFD93D'  # 黄色\n",
        "\n",
        "# 绘制输入\n",
        "vision_input = FancyBboxPatch((0.5, 4), 1.5, 1, \n",
        "                               boxstyle=\"round,pad=0.1\", \n",
        "                               facecolor=color_vision, \n",
        "                               edgecolor='black', linewidth=2)\n",
        "ax.add_patch(vision_input)\n",
        "ax.text(1.25, 4.5, '视觉输入\\n(图像)', ha='center', va='center', \n",
        "        fontsize=12, weight='bold')\n",
        "\n",
        "language_input = FancyBboxPatch((0.5, 2), 1.5, 1, \n",
        "                                boxstyle=\"round,pad=0.1\", \n",
        "                                facecolor=color_language, \n",
        "                                edgecolor='black', linewidth=2)\n",
        "ax.add_patch(language_input)\n",
        "ax.text(1.25, 2.5, '语言输入\\n(文本)', ha='center', va='center', \n",
        "        fontsize=12, weight='bold')\n",
        "\n",
        "# 绘制编码器\n",
        "vision_encoder = FancyBboxPatch((3, 4), 1.5, 1, \n",
        "                                boxstyle=\"round,pad=0.1\", \n",
        "                                facecolor=color_vision, \n",
        "                                edgecolor='black', linewidth=2)\n",
        "ax.add_patch(vision_encoder)\n",
        "ax.text(3.75, 4.5, '视觉编码器', ha='center', va='center', \n",
        "        fontsize=11, weight='bold')\n",
        "\n",
        "language_encoder = FancyBboxPatch((3, 2), 1.5, 1, \n",
        "                                  boxstyle=\"round,pad=0.1\", \n",
        "                                  facecolor=color_language, \n",
        "                                  edgecolor='black', linewidth=2)\n",
        "ax.add_patch(language_encoder)\n",
        "ax.text(3.75, 2.5, '语言编码器', ha='center', va='center', \n",
        "        fontsize=11, weight='bold')\n",
        "\n",
        "# 绘制融合模块\n",
        "fusion = FancyBboxPatch((5.5, 2.5), 1.5, 1, \n",
        "                        boxstyle=\"round,pad=0.1\", \n",
        "                        facecolor=color_fusion, \n",
        "                        edgecolor='black', linewidth=2)\n",
        "ax.add_patch(fusion)\n",
        "ax.text(6.25, 3, '多模态融合', ha='center', va='center', \n",
        "        fontsize=11, weight='bold')\n",
        "\n",
        "# 绘制动作解码器\n",
        "action_decoder = FancyBboxPatch((8, 2.5), 1.5, 1, \n",
        "                                boxstyle=\"round,pad=0.1\", \n",
        "                                facecolor=color_action, \n",
        "                                edgecolor='black', linewidth=2)\n",
        "ax.add_patch(action_decoder)\n",
        "ax.text(8.75, 3, '动作解码器', ha='center', va='center', \n",
        "        fontsize=11, weight='bold')\n",
        "\n",
        "# 绘制输出\n",
        "action_output = FancyBboxPatch((8, 0.5), 1.5, 1, \n",
        "                                boxstyle=\"round,pad=0.1\", \n",
        "                                facecolor=color_action, \n",
        "                                edgecolor='black', linewidth=2)\n",
        "ax.add_patch(action_output)\n",
        "ax.text(8.75, 1, '动作输出', ha='center', va='center', \n",
        "        fontsize=12, weight='bold')\n",
        "\n",
        "# 绘制箭头\n",
        "arrow1 = FancyArrowPatch((2, 4.5), (3, 4.5), \n",
        "                         arrowstyle='->', lw=2, color='black')\n",
        "ax.add_patch(arrow1)\n",
        "arrow2 = FancyArrowPatch((2, 2.5), (3, 2.5), \n",
        "                         arrowstyle='->', lw=2, color='black')\n",
        "ax.add_patch(arrow2)\n",
        "arrow3 = FancyArrowPatch((4.5, 4.5), (5.5, 3.25), \n",
        "                         arrowstyle='->', lw=2, color='black')\n",
        "ax.add_patch(arrow3)\n",
        "arrow4 = FancyArrowPatch((4.5, 2.5), (5.5, 2.75), \n",
        "                         arrowstyle='->', lw=2, color='black')\n",
        "ax.add_patch(arrow4)\n",
        "arrow5 = FancyArrowPatch((7, 3), (8, 3), \n",
        "                         arrowstyle='->', lw=2, color='black')\n",
        "ax.add_patch(arrow5)\n",
        "arrow6 = FancyArrowPatch((8.75, 2.5), (8.75, 1.5), \n",
        "                         arrowstyle='->', lw=2, color='black')\n",
        "ax.add_patch(arrow6)\n",
        "\n",
        "# 添加标题\n",
        "ax.text(5, 5.5, 'VLA模型架构', ha='center', va='center', \n",
        "        fontsize=16, weight='bold')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"=\" * 60)\n",
        "print(\"VLA模型架构说明：\")\n",
        "print(\"=\" * 60)\n",
        "print(\"1. 视觉输入和语言输入分别进入视觉编码器和语言编码器\")\n",
        "print(\"2. 两个编码器提取特征后，进入多模态融合模块\")\n",
        "print(\"3. 融合后的特征进入动作解码器，生成动作序列\")\n",
        "print(\"4. 整个流程是端到端的，所有模块统一优化\")\n",
        "print(\"=\" * 60)"
    ]
}

notebook["cells"].append(cell4)

# 保存notebook
output_path = r"f:\大学本科\人工智能学习\科研_VLA\学习文档\00_VLA全景导览\01_VLA是什么_现状与未来\理论笔记\VLA基本概念详解.ipynb"

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=2)

print(f"Notebook已创建：{output_path}")
print("注意：这是一个基础版本，后续可以继续补充更多内容（如完整的术语表、数学推导、更多可视化等）")


