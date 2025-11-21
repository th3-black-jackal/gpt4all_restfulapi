from typing import List, Dict, Any, Optional
import re

from gpt4all import GPT4All, Embed4All
import PyPDF2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import hashlib



class PDFProcessor:
    """Handles PDF text extraction and processing"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""

    @staticmethod
    def extract_github_links(text: str) -> List[Dict[str, str]]:
        """Extract GitHub URLs from text"""
        github_pattern = r'https?://github\.com/[\w\-\.]+/[\w\-\.]+'
        links = re.findall(github_pattern, text)
        
        repositories = []
        for link in links:
            repo_type = "repository"
            if "/issues/" in link:
                repo_type = "issue"
            elif "/pull/" in link:
                repo_type = "pull_request"
            elif "/gist/" in link:
                repo_type = "gist"
            elif re.match(r'https?://github\.com/[\w\-\.]+$', link):
                repo_type = "profile"
                
            repositories.append({
                "url": link,
                "type": repo_type
            })
        
        return repositories



class EmbeddingGenerator:
    """Generates embeddings for text using GPT4All"""
    
    def __init__(self, model: GPT4All):
        self.model = model
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        # For GPT4All, we'll use a simple approach to generate embeddings
        # In a real implementation, you might want to use a dedicated embedding model
        truncated_text = text[:1000]  # Limit text length
        embedder = Embed4All(device='gpu')
        embedding = embedder.embed(truncated_text)
        return embedding

class TopicClusterer:
    """Clusters papers into topics based on embeddings"""
    
    def __init__(self):
        self.dbscan = DBSCAN(eps=0.5, min_samples=2, metric='cosine')
    
    def cluster_papers(self, embeddings: List[List[float]]) -> List[int]:
        """Cluster papers based on embeddings"""
        if len(embeddings) < 2:
            return [0] * len(embeddings)
        
        similarity_matrix = cosine_similarity(embeddings)
        distance_matrix = 1 - similarity_matrix
        
        clusters = self.dbscan.fit_predict(distance_matrix)
        return clusters.tolist()




class GapAnalyzer:
    """Analyzes gaps between research papers"""
    
    def __init__(self, model: GPT4All):
        self.model = model
    
    def analyze_gaps(self, papers: List[Dict], clusters: List[int]) -> List[Dict]:
        """Analyze gaps between papers in the same cluster"""
        gaps = []
        
        # Group papers by cluster
        cluster_groups = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(papers[i])
        
        # Analyze gaps for each cluster
        for cluster_id, cluster_papers in cluster_groups.items():
            if len(cluster_papers) < 2:
                continue
                
            cluster_gaps = self._analyze_cluster_gaps(cluster_papers, cluster_id)
            gaps.extend(cluster_gaps)
        
        return gaps
    
    def _analyze_cluster_gaps(self, papers: List[Dict], cluster_id: int) -> List[Dict]:
        """Analyze gaps for a specific cluster"""
        try:
            # Create context from papers
            context = ""
            for i, paper in enumerate(papers):
                context += f"Paper {i+1}:\n"
                context += f"Title: {paper.get('title', 'Unknown')}\n"
                context += f"Content: {paper.get('content', '')[:2000]}\n\n"
            
            prompt = f"""Analyze these research papers and identify research gaps:

{context}

Identify gaps in these areas:
1. Datasets used
2. Methods employed
3. Evaluation metrics
4. Baselines compared
5. Limitations mentioned
6. Reproducibility artifacts

For each gap, provide:
- A clear gap statement
- Evidence from the papers
- Impact level (High/Medium/Low)
- Suggested next steps

Format as JSON list with keys: statement, evidence, impact, next_steps, papers_involved"""

            response = self.model.generate(prompt, max_tokens=2000)
            
            # Parse JSON response
            gaps = self._parse_gap_response(response)
            
            # Add cluster information
            for gap in gaps:
                gap['cluster_id'] = cluster_id
                gap['papers_involved'] = [p.get('title', f'Paper {i+1}') for i, p in enumerate(papers)]
            
            return gaps
            
        except Exception as e:
            print(f"Error analyzing gaps: {e}")
            return []
    
    def _parse_gap_response(self, response: str) -> List[Dict]:
        """Parse GPT4All response into gap structures"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return []
        except:
            return []


class SummaryGenerator:
    """Generates summaries for research papers"""
    
    def __init__(self, model: GPT4All, max_tokens: int = 200):
        self.model = model
        self.max_tokens = max_tokens
    
    def generate_summary(self, paper_content: str, paper_title: str = "") -> str:
        """Generate a 150-word technical summary"""
        prompt = f"""Generate a concise 150-word technical summary of this research paper.

Title: {paper_title}
Content: {paper_content[:3000]}

Requirements:
- Abstractive summary (not extractive)
- Exactly 150 words
- Technical audience
- No citations, graphs, or references
- Focus on key contributions and methods

Summary:"""

        try:
            summary = self.model.generate(prompt, max_tokens=self.max_tokens)
            # Clean up summary
            summary = re.sub(r'\[.*?\]', '', summary)  # Remove citations
            summary = re.sub(r'\(.*?\)', '', summary)  # Remove parentheses
            summary = ' '.join(summary.split())  # Normalize whitespace
            
            # Ensure ~150 words
            words = summary.split()
            print(f"Created summary of: {len(words)} words.")
            if len(words) > 160:
                summary = ' '.join(words[:150])
            elif len(words) < 140:
                # Regenerate if too short
                summary = self.model.generate(prompt + " Make it more detailed.", max_tokens=self.max_tokens)
            
            return summary.strip()
        except Exception as e:
            return f"Error generating summary: {e}"
