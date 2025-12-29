import os
from pathlib import Path
from datetime import datetime

# 要检查的文件夹名称
target_folders = ["代码示例", "练习题"]

# 要搜索的根目录
base_dir = "科研_VLA/学习文档"

def find_folders(root_dir, folder_names):
    """递归查找指定名称的文件夹"""
    found_folders = {name: [] for name in folder_names}
    
    if not os.path.exists(root_dir):
        print(f"警告: 目录不存在 - {root_dir}")
        return found_folders
    
    # 遍历所有子目录
    for root, dirs, files in os.walk(root_dir):
        for dir_name in dirs:
            if dir_name in folder_names:
                folder_path = os.path.join(root, dir_name)
                # 转换为相对路径
                rel_path = os.path.relpath(folder_path, root_dir)
                found_folders[dir_name].append(rel_path)
    
    return found_folders

def generate_markdown_report(found_folders, base_dir):
    """生成Markdown报告"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    markdown = f"""# 文件夹检查报告

**检查时间**: {timestamp}  
**检查目录**: `{base_dir}`  
**目标文件夹**: `代码示例`, `练习题`

---

## 检查结果概览

"""
    
    total_count = sum(len(folders) for folders in found_folders.values())
    
    if total_count == 0:
        markdown += "[成功] **所有目标文件夹已成功删除！**\n\n"
        markdown += "未发现任何 `代码示例` 或 `练习题` 文件夹。\n"
    else:
        markdown += f"[警告] **发现 {total_count} 个目标文件夹仍存在**\n\n"
        
        for folder_name in target_folders:
            count = len(found_folders[folder_name])
            if count > 0:
                markdown += f"- `{folder_name}`: **{count} 个**\n"
            else:
                markdown += f"- `{folder_name}`: [已删除] **0 个** (已全部删除)\n"
    
    markdown += "\n---\n\n## 详细列表\n\n"
    
    # 按文件夹类型分组显示
    for folder_name in target_folders:
        folders = found_folders[folder_name]
        if folders:
            markdown += f"### {folder_name} ({len(folders)} 个)\n\n"
            for i, folder_path in enumerate(sorted(folders), 1):
                markdown += f"{i}. `{folder_path}`\n"
            markdown += "\n"
        else:
            markdown += f"### {folder_name}\n\n[已删除] 未发现任何 `{folder_name}` 文件夹。\n\n"
    
    # 添加完整目录结构
    markdown += "---\n\n## 完整目录结构（包含所有叶子节点）\n\n"
    markdown += "以下为 `学习文档` 目录的完整结构，递归到所有层级：\n\n"
    
    def get_structure(path, prefix="", exclude_dirs=None):
        """获取完整的目录结构（递归到所有叶子节点）"""
        if exclude_dirs is None:
            exclude_dirs = set(target_folders)  # 排除已删除的目标文件夹
        
        structure = ""
        try:
            items = sorted(os.listdir(path))
            dirs = []
            files = []
            
            for item in items:
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    if not item.startswith('.') and item not in exclude_dirs:
                        dirs.append(item)
                elif os.path.isfile(item_path):
                    if not item.startswith('.'):
                        files.append(item)
            
            # 先显示文件
            all_items = files + dirs
            for i, item in enumerate(all_items):
                is_last = (i == len(all_items) - 1)
                current_prefix = "└── " if is_last else "├── "
                item_path = os.path.join(path, item)
                
                # 判断是文件还是目录
                if os.path.isfile(item_path):
                    structure += f"{prefix}{current_prefix}{item} (文件)\n"
                else:
                    structure += f"{prefix}{current_prefix}{item}/\n"
                    
                    # 递归处理子目录
                    next_prefix = prefix + ("    " if is_last else "│   ")
                    structure += get_structure(item_path, next_prefix, exclude_dirs)
        except (PermissionError, OSError) as e:
            structure += f"{prefix}    [无法访问: {str(e)}]\n"
        
        return structure
    
    structure = get_structure(base_dir)
    markdown += "```\n"
    markdown += base_dir + "/\n"
    markdown += structure
    markdown += "```\n"
    
    # 添加统计信息
    def count_items(path, exclude_dirs=None):
        """统计目录和文件数量"""
        if exclude_dirs is None:
            exclude_dirs = set(target_folders)
        
        dir_count = 0
        file_count = 0
        
        try:
            for root, dirs, files in os.walk(path):
                # 过滤排除的目录
                dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]
                files = [f for f in files if not f.startswith('.')]
                
                dir_count += len(dirs)
                file_count += len(files)
        except (PermissionError, OSError):
            pass
        
        return dir_count, file_count
    
    dir_count, file_count = count_items(base_dir)
    markdown += f"\n**统计信息**:\n"
    markdown += f"- 目录总数: {dir_count}\n"
    markdown += f"- 文件总数: {file_count}\n"
    
    return markdown

def main():
    print("开始检查目录结构...")
    print("=" * 60)
    
    # 查找文件夹
    found_folders = find_folders(base_dir, target_folders)
    
    # 生成报告
    markdown_content = generate_markdown_report(found_folders, base_dir)
    
    # 保存到文件
    output_file = "文件夹检查报告.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"\n检查完成！")
    print(f"报告已保存到: {output_file}")
    
    # 打印摘要
    total = sum(len(folders) for folders in found_folders.values())
    if total == 0:
        print("\n[成功] 所有目标文件夹已成功删除！")
    else:
        print(f"\n[警告] 发现 {total} 个目标文件夹仍存在：")
        for folder_name in target_folders:
            count = len(found_folders[folder_name])
            if count > 0:
                print(f"  - {folder_name}: {count} 个")
                for path in found_folders[folder_name][:5]:  # 只显示前5个
                    print(f"    - {path}")
                if len(found_folders[folder_name]) > 5:
                    print(f"    ... 还有 {len(found_folders[folder_name]) - 5} 个")

if __name__ == "__main__":
    main()

