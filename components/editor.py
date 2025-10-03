import re
from typing import Optional, List, Any, Dict,Union


class CitationChecker:
    """判断参考文献是否在内容中被正确引用的工具类"""

    def __init__(self):
        # 用于识别一些常见的引用模式
        self.citation_patterns = [
            # 中文引用格式：姓名等（年份）
            r'([^\s,.;，。；]+)等(?:人|)[\s]*[（(](\d{4})[）)]',
            # 中文引用格式：（姓名等，年份）
            r'[（(]([^\s,.;，。；]+)等[，,][\s]*(\d{4})[）)]',
            # 西文引用格式：Name et al. (year)
            r'([^\s,.;]+)[\s]+et[\s]*al\.[\s]*[（(](\d{4})[）)]',
            # 西文引用格式：(Name et al., year)
            r'[（(]([^\s,.;]+)[\s]+et[\s]*al\.[\s]*[,，][\s]*(\d{4})[）)]',
            # 常规引用格式：(Author & Author, year)
            r'[（(]([^\s,.;&]+)[\s]*&[\s]*([^\s,.;&]+)[\s]*[,，][\s]*(\d{4})[）)]',
            # 常规引用格式：Author and Author (year)
            r'([^\s,.;&]+)[\s]+(?:and|与|和)[\s]+([^\s,.;&]+)[\s]*[（(](\d{4})[）)]'
        ]

    def normalize_text(self, text):
        """标准化文本，处理标点符号和空格"""
        # 统一标点符号（中文转英文）
        text = text.replace("。", ".").replace("；", ";")
        text = text.replace("？", "?").replace("！", "!").replace("…", "...")
        text = text.replace("，", ",").replace("（", "(").replace("）", ")")

        # 标准化空格
        text = re.sub(r'\s+', ' ', text)

        return text

    def extract_citation_info(self, citation):
        """从引用文本中提取作者和年份信息"""
        citation = self.normalize_text(citation)

        # 去掉外部括号（如果有）
        if citation.startswith("(") and citation.endswith(")"):
            citation = citation[1:-1]

        # 提取所有作者
        authors = []

        # 处理多种分隔符
        if "&" in citation:
            # 处理 Author & Author, year 格式
            parts = citation.split(",")
            author_part = parts[0].strip()
            author_names = [a.strip() for a in author_part.split("&")]
            authors.extend(author_names)

            # 提取年份
            year = None
            for part in parts[1:]:
                year_match = re.search(r'\d{4}', part)
                if year_match:
                    year = year_match.group(0)
                    break

        elif "et al." in citation:
            # 处理 Author et al., year 格式
            parts = citation.split("et al.")
            if parts[0].strip():
                authors.append(parts[0].strip())

            # 添加et al.作为标识符
            authors.append("et al")

            # 提取年份
            year = None
            year_match = re.search(r'\d{4}', parts[1])
            if year_match:
                year = year_match.group(0)

        else:
            # 处理简单格式或未识别格式
            parts = citation.split(",")

            # 假设第一部分是作者，最后匹配到的4位数字是年份
            if len(parts) > 0:
                authors.append(parts[0].strip())

            # 查找年份
            year = None
            year_match = re.search(r'\d{4}', citation)
            if year_match:
                year = year_match.group(0)

        return {"authors": authors, "year": year}

    def split_sentences(self, content):
        """分割内容为句子"""
        content = self.normalize_text(content)

        # 使用正则表达式找到句子边界
        # 这里考虑了英文句号、问号、感叹号和分号作为句子分隔符
        sentences = re.split(r'(?<=[.!?;])\s+', content)

        # 处理最后一个句子可能没有标点符号的情况
        sentences = [s.strip() for s in sentences if s.strip()]

        # 对句子进行后处理，处理引号内的句号不分割句子的情况
        processed_sentences = []
        current = ""

        for sentence in sentences:
            if current:
                current += " " + sentence
            else:
                current = sentence

            # 引号配对检查
            if current.count('"') % 2 == 0 and current.count("'") % 2 == 0:
                processed_sentences.append(current)
                current = ""

        if current:
            processed_sentences.append(current)

        return processed_sentences

    def is_citation_in_content(self, citation, content):
        """判断引用是否在内容中被正确使用"""
        # 提取引用信息
        citation_info = self.extract_citation_info(citation)
        authors = citation_info["authors"]
        year = citation_info["year"]

        if not authors or not year:
            return False

        # 分割内容为句子
        sentences = self.split_sentences(content)

        # 检查每个句子是否包含所有引用元素
        for sentence in sentences:
            sentence_lower = sentence.lower()
            all_elements_present = True

            # 检查所有作者是否在句子中
            for author in authors:
                if author.lower() not in sentence_lower:
                    # 特殊处理 "et al" 变体
                    if author.lower() == "et al":
                        if not any(variant in sentence_lower for variant in ["et al", "等人", "等"]):
                            all_elements_present = False
                            break
                    else:
                        all_elements_present = False
                        break

            # 检查年份是否在句子中
            if year not in sentence:
                all_elements_present = False

            if all_elements_present:
                return True

        # 使用正则表达式模式直接匹配常见引用格式
        content_normalized = self.normalize_text(content)
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, content_normalized)
            if matches:
                # 根据匹配模式检查作者和年份
                for match in matches:
                    if isinstance(match, tuple):
                        # 对于多组捕获的模式，检查最后一个元素是否为年份
                        matched_year = match[-1]
                        if matched_year == year:
                            # 检查作者是否匹配
                            for author in authors:
                                if author != "et al":
                                    found = False
                                    for item in match[:-1]:
                                        if author.lower() in item.lower():
                                            found = True
                                            break
                                    if not found:
                                        break
                            else:
                                return True

        return False


def format_search_results_apa(sentence_results: Dict[str, List[Dict[str, Any]]]) -> List[Dict]:
    """
    将搜索结果转换为APA格式的引用和参考文献，按句子分组为列表

    Args:
        sentence_results: 句子到搜索结果的映射

    Returns:
        包含句子和文献列表的字典列表 [{"sentence": str, "literature": list[dict]}, ...]
    """
    result_list = []

    for sentence, results in sentence_results.items():
        literature_list = []

        for item in results:
            # 提取必要信息
            authors = item.get('authors', '')
            year = item.get('year', '')
            title = item.get('title', '')
            journal = item.get('journal', '')
            content = item.get('content_with_weight', '')

            # 提取额外字段
            volume = item.get('vO', '')  # 卷号
            issue = item.get('issue', '')  # 期号
            pages = item.get('page_range', '')  # 页码范围
            pdf_url=item.get('pdf_url','')

            # 处理作者
            author_list = authors.split(";")

            # 处理作者名称格式
            processed_authors = []
            for author in author_list:
                author = author.strip()
                if author:
                    # 检查是否有逗号（可能是英文格式：姓,名）
                    if "," in author:
                        processed_authors.append(author.split(",")[0].strip())
                    else:
                        processed_authors.append(author)

            # 根据作者数量格式化引用
            if len(processed_authors) == 0:
                formatted_author = ""
            elif len(processed_authors) == 1:
                formatted_author = processed_authors[0]
            elif len(processed_authors) == 2:
                formatted_author = f"{processed_authors[0]} & {processed_authors[1]}"
            else:  # 3个或更多作者
                formatted_author = f"{processed_authors[0]} et al."

            # 创建APA引用 (作者, 年份)
            apa_citation = f"({formatted_author}, {year})" if formatted_author and year else ""

            # 创建带引用的内容
            content_with_citation = f"{content} {apa_citation}" if content else apa_citation

            # 创建完整的APA参考文献
            # 对于参考文献，我们会使用所有作者
            if len(processed_authors) == 0:
                authors_for_reference = ""
            elif len(processed_authors) == 1:
                authors_for_reference = processed_authors[0]
            elif len(processed_authors) == 2:
                authors_for_reference = f"{processed_authors[0]} & {processed_authors[1]}"
            else:
                # 在完整参考中，APA使用最多20位作者再使用et al.
                # 为简单起见，我们将包括所有作者
                authors_for_reference = ", ".join(processed_authors[:-1]) + ", & " + processed_authors[-1]

            # 构建参考文献，包括卷、期和页码（如果有）
            apa_reference = f"{authors_for_reference}. ({year}). {title}. {journal}"

            # 添加卷号（如果有）
            if volume:
                apa_reference += f", {volume}"

                # 添加期号（仅当卷号也存在时）
                if issue:
                    apa_reference += f"({issue})"

            # 添加页码（如果有）
            if pages:
                apa_reference += f", {pages}"

            # 添加结束句点
            apa_reference += "."
            apa_reference+= f"[PDF_URL_START]{pdf_url}[PDF_URL_END]"

            # 添加到文献列表
            literature_list.append({
                'apa_citation': apa_citation,
                'content_with_citation': content_with_citation,
                'apa_reference': apa_reference
            })

        # 添加句子和对应的文献列表到结果
        result_list.append({
            'sentence': sentence,
            'literature': literature_list
        })

    return result_list

def extract_valid_references(text: str, literature_list: List[Dict], citation_checker) -> List[str]:
    """
    从文本中提取有效的引用，并返回对应的参考文献列表

    Args:
        text: 需要检查的文本内容
        literature_list: 文献列表，格式为[{'sentence': str, 'literature': list[dict]}, ...]
        citation_checker: CitationChecker实例，用于验证引用

    Returns:
        有效引用的参考文献列表
    """
    # 用于收集有效引用的参考文献
    valid_references = set()

    # 处理每个句子及其对应的文献
    for item in literature_list:
        literature = item.get('literature', [])

        # 处理每篇文献
        for lit in literature:
            citation = lit.get('apa_citation', '')
            reference = lit.get('apa_reference', '')

            # 使用CitationChecker验证引用是否在文本中
            if citation and citation_checker.is_citation_in_content(citation, text.lower()):
                valid_references.add(reference)
                continue

            # 尝试构造中文引用形式（如"赵亚普等，2024"）并验证
            if '(' in citation and ')' in citation:
                try:
                    # 提取第一个作者
                    author_part = citation.split('(')[1].split(',')[0].strip()
                    # 处理"et al."
                    if "et al." in author_part:
                        author = author_part.split("et al.")[0].strip()
                        # 提取年份
                        year_part = citation.split(',')[-1].split(')')[0].strip()
                        # 构造中文引用形式
                        chinese_citation = f"{author}等，{year_part}"
                        if citation_checker.is_citation_in_content(chinese_citation, text.lower()):
                            valid_references.add(reference)
                except:
                    continue

    # 将集合转换为列表并返回
    return sorted(list(valid_references))

citation_checker = CitationChecker()
