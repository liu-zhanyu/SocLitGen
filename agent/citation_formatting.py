"""
Citation Formatting Module

This module handles APA citation formatting and citation matching. It transforms literature
metadata into proper APA format and checks if citations are actually used in the text.
"""

from typing import List, Dict
from components.editor import citation_checker


class CitationFormattingAgent:
    """
    Citation Formatting Agent for APA format conversion and citation matching.
    """

    def __init__(self):
        self.name = "Citation Formatting Agent"

    def transform_to_apa_format(self, data_list: List[Dict]) -> List[Dict]:
        """
        Transform literature metadata to APA format.

        Args:
            data_list: List of literature dictionaries with metadata

        Returns:
            List of dictionaries with APA-formatted citations and references
        """
        result_list = []

        for item in data_list:
            result_item = {}

            # Extract metadata
            authors = item.get('authors', '')
            year = item.get('year', '')
            title = item.get('title', '')
            journal = item.get('journal', '')
            content = item.get('content_with_weight', '')

            # Additional fields for full reference
            volume = item.get('vO', '')
            issue = item.get('issue', '')
            pages = item.get('page_range', '')
            pdf_url = item.get('pdf_url', '')

            # Process authors for citation
            author_list = authors.split(";")
            processed_authors = []

            for author in author_list:
                author = author.strip()
                if author:
                    if "," in author:
                        processed_authors.append(author.split(",")[0].strip())
                    else:
                        processed_authors.append(author)

            # Format in-text citation
            if len(processed_authors) == 0:
                formatted_author = ""
            elif len(processed_authors) == 1:
                formatted_author = processed_authors[0]
            elif len(processed_authors) == 2:
                formatted_author = f"{processed_authors[0]} & {processed_authors[1]}"
            else:
                formatted_author = f"{processed_authors[0]} et al."

            # Create APA in-text citation
            apa_citation = f"({formatted_author}, {year})"
            content_with_citation = f"{content} {apa_citation}"

            # Create full APA reference
            if len(processed_authors) == 0:
                authors_for_reference = ""
            elif len(processed_authors) == 1:
                authors_for_reference = processed_authors[0]
            elif len(processed_authors) == 2:
                authors_for_reference = f"{processed_authors[0]} & {processed_authors[1]}"
            else:
                authors_for_reference = ", ".join(processed_authors[:-1]) + ", & " + processed_authors[-1]

            # Build full reference
            apa_reference = f"{authors_for_reference}. ({year}). {title}. {journal}"

            if volume:
                apa_reference += f", {volume}"
                if issue:
                    apa_reference += f"({issue})"

            if pages:
                apa_reference += f", {pages}"

            apa_reference += "."
            apa_reference += f"[PDF_URL_START]{pdf_url}[PDF_URL_END]"

            # Store results
            result_item['apa_citation'] = apa_citation
            result_item['content_with_citation'] = content_with_citation
            result_item['original_content'] = content
            result_item['apa_reference'] = apa_reference

            result_list.append(result_item)

        return result_list

    def is_citation_in_content(self, citation: str, content: str) -> bool:
        """
        Check if a citation appears in the content text.

        Args:
            citation: Citation string in (Author, Year) format
            content: Text content to search in

        Returns:
            True if citation is found in content, False otherwise
        """
        return citation_checker.is_citation_in_content(citation, content)

    def extract_cited_references(
        self,
        text: str,
        literature_data: List[Dict]
    ) -> List[str]:
        """
        Extract references that are actually cited in the text.

        Args:
            text: Full text content
            literature_data: List of literature items with metadata

        Returns:
            List of APA-formatted reference strings that appear in text
        """
        cited_references = set()

        # Transform to APA format
        apa_formatted = self.transform_to_apa_format(literature_data)

        # Check each citation
        for item in apa_formatted:
            citation = item.get("apa_citation", "")
            if citation and self.is_citation_in_content(citation, text.lower()):
                cited_references.add(item["apa_reference"])

        return list(cited_references)