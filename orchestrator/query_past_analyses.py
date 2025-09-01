#!/usr/bin/env python3
"""
Query and retrieve past AIOS analyses and thought streams
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from aios_strategic_orchestrator import AIOSStrategicOrchestrator

class AIOSAnalysisRetriever:
    """Tool for accessing and querying past AIOS analyses"""
    
    def __init__(self):
        self.orchestrator = AIOSStrategicOrchestrator()
        self.results_dir = os.path.dirname(os.path.abspath(__file__))
    
    async def search_by_keywords(self, keywords: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """Search for past analyses by keywords"""
        print(f"🔍 Searching for analyses containing: {', '.join(keywords)}")
        
        results = []
        for keyword in keywords:
            search_results = await self.orchestrator.search_thought_streams(
                agent_name=None,  # Search across all agents
                semantic_query=keyword,
                limit=limit
            )
            if search_results.get("streams"):
                results.extend(search_results["streams"])
        
        # Deduplicate results
        seen_ids = set()
        unique_results = []
        for result in results:
            if result["stream_id"] not in seen_ids:
                seen_ids.add(result["stream_id"])
                unique_results.append(result)
        
        return unique_results
    
    async def search_by_agent(self, agent_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find all analyses performed by a specific agent"""
        print(f"👤 Finding analyses by agent: {agent_name}")
        
        results = await self.orchestrator.search_thought_streams(
            agent_name=agent_name,
            limit=limit
        )
        
        return results.get("streams", [])
    
    async def search_by_cognitive_type(self, cognitive_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find analyses of a specific cognitive type"""
        print(f"🧠 Finding {cognitive_type} analyses")
        
        results = await self.orchestrator.search_thought_streams(
            cognitive_type=cognitive_type,
            limit=limit
        )
        
        return results.get("streams", [])
    
    async def get_recent_analyses(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get all analyses from the past N hours"""
        print(f"📅 Getting analyses from the past {hours} hours")
        
        # Search for all recent streams
        results = await self.orchestrator.search_thought_streams(
            limit=100  # Get more results to filter by time
        )
        
        # Filter by time (would need timestamp in actual implementation)
        return results.get("streams", [])
    
    def load_saved_analysis(self, filename: str) -> Dict[str, Any]:
        """Load a specific saved analysis file"""
        filepath = os.path.join(self.results_dir, filename)
        
        if not os.path.exists(filepath):
            return {"error": f"File not found: {filename}"}
        
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def list_saved_files(self, pattern: str = "*.json") -> List[str]:
        """List all saved analysis files"""
        import glob
        
        files = glob.glob(os.path.join(self.results_dir, pattern))
        return [os.path.basename(f) for f in files]
    
    async def get_thought_stream_details(self, stream_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific thought stream"""
        print(f"📊 Getting details for stream: {stream_id}")
        
        # Access the stream directly from AIOS memory
        if stream_id in self.orchestrator.thought_streams:
            stream = self.orchestrator.thought_streams[stream_id]
            return {
                "stream_id": stream_id,
                "agents": list(stream["agents"]),
                "thoughts": stream["thoughts"],
                "cognitive_state": stream["cognitive_state"],
                "attention_level": stream["attention_level"],
                "created_at": stream.get("created_at", "Unknown"),
                "last_updated": stream.get("last_updated", "Unknown")
            }
        else:
            return {"error": f"Stream {stream_id} not found"}
    
    async def export_analysis(self, stream_id: str, format: str = "json") -> str:
        """Export an analysis in different formats"""
        details = await self.get_thought_stream_details(stream_id)
        
        if "error" in details:
            return details["error"]
        
        if format == "json":
            return json.dumps(details, indent=2)
        elif format == "markdown":
            md = f"# Thought Stream Analysis: {stream_id}\n\n"
            md += f"## Metadata\n"
            md += f"- **Created**: {details.get('created_at', 'Unknown')}\n"
            md += f"- **Last Updated**: {details.get('last_updated', 'Unknown')}\n"
            md += f"- **Cognitive State**: {details.get('cognitive_state', 'Unknown')}\n"
            md += f"- **Attention Level**: {details.get('attention_level', 0)}\n\n"
            
            md += f"## Participating Agents\n"
            for agent in details.get('agents', []):
                md += f"- {agent}\n"
            
            md += f"\n## Thoughts\n"
            for i, thought in enumerate(details.get('thoughts', []), 1):
                md += f"\n### Thought {i}\n"
                md += f"**Agent**: {thought.get('agent_name', 'Unknown')}\n\n"
                md += f"{thought.get('content', '')}\n"
            
            return md
        else:
            return f"Unsupported format: {format}"
    
    async def create_analysis_report(self) -> Dict[str, Any]:
        """Create a comprehensive report of all analyses"""
        print("📈 Creating comprehensive analysis report...")
        
        # Get all available streams
        all_streams = await self.orchestrator.search_thought_streams(limit=100)
        
        # Get saved files
        saved_files = self.list_saved_files()
        
        # Compile statistics
        report = {
            "timestamp": datetime.now().isoformat(),
            "statistics": {
                "total_thought_streams": len(all_streams.get("streams", [])),
                "saved_analysis_files": len(saved_files),
                "active_streams": len(self.orchestrator.thought_streams),
                "total_agents_used": len(set(
                    agent for stream in all_streams.get("streams", [])
                    for agent in stream.get("agents", [])
                ))
            },
            "recent_analyses": all_streams.get("streams", [])[:5],
            "saved_files": saved_files,
            "cognitive_types_used": {},
            "agent_participation": {}
        }
        
        # Analyze cognitive types and agent participation
        for stream in all_streams.get("streams", []):
            # Count cognitive types
            cog_type = stream.get("cognitive_type", "unknown")
            report["cognitive_types_used"][cog_type] = report["cognitive_types_used"].get(cog_type, 0) + 1
            
            # Count agent participation
            for agent in stream.get("agents", []):
                report["agent_participation"][agent] = report["agent_participation"].get(agent, 0) + 1
        
        return report

async def interactive_query():
    """Interactive command-line interface for querying analyses"""
    retriever = AIOSAnalysisRetriever()
    
    print("\n🤖 AIOS Analysis Query Tool")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Search by keywords")
        print("2. Search by agent name")
        print("3. Search by cognitive type")
        print("4. Get recent analyses")
        print("5. Load saved file")
        print("6. Get stream details")
        print("7. Create full report")
        print("8. Export analysis")
        print("9. Exit")
        
        choice = input("\nSelect option (1-9): ").strip()
        
        if choice == "1":
            keywords = input("Enter keywords (comma-separated): ").split(",")
            keywords = [k.strip() for k in keywords]
            results = await retriever.search_by_keywords(keywords)
            print(f"\nFound {len(results)} results:")
            for r in results:
                print(f"  - {r['stream_id']}: {r.get('description', 'No description')}")
        
        elif choice == "2":
            agent = input("Enter agent name: ").strip()
            results = await retriever.search_by_agent(agent)
            print(f"\nFound {len(results)} analyses by {agent}")
        
        elif choice == "3":
            cog_type = input("Enter cognitive type (reasoning/analysis/synthesis/etc): ").strip()
            results = await retriever.search_by_cognitive_type(cog_type)
            print(f"\nFound {len(results)} {cog_type} analyses")
        
        elif choice == "4":
            hours = int(input("Past how many hours? (default 24): ") or "24")
            results = await retriever.get_recent_analyses(hours)
            print(f"\nFound {len(results)} recent analyses")
        
        elif choice == "5":
            files = retriever.list_saved_files()
            print("\nAvailable files:")
            for i, f in enumerate(files, 1):
                print(f"  {i}. {f}")
            if files:
                idx = int(input("Select file number: ")) - 1
                if 0 <= idx < len(files):
                    data = retriever.load_saved_analysis(files[idx])
                    print(json.dumps(data, indent=2)[:500] + "...")
        
        elif choice == "6":
            stream_id = input("Enter stream ID: ").strip()
            details = await retriever.get_thought_stream_details(stream_id)
            print(json.dumps(details, indent=2))
        
        elif choice == "7":
            report = await retriever.create_analysis_report()
            print("\n📊 Analysis Report:")
            print(f"Total Streams: {report['statistics']['total_thought_streams']}")
            print(f"Saved Files: {report['statistics']['saved_analysis_files']}")
            print(f"Active Streams: {report['statistics']['active_streams']}")
            print(f"Unique Agents: {report['statistics']['total_agents_used']}")
            
            print("\nTop Agents:")
            for agent, count in sorted(report['agent_participation'].items(), 
                                      key=lambda x: x[1], reverse=True)[:5]:
                print(f"  - {agent}: {count} participations")
        
        elif choice == "8":
            stream_id = input("Enter stream ID to export: ").strip()
            format = input("Format (json/markdown): ").strip() or "json"
            export = await retriever.export_analysis(stream_id, format)
            
            filename = f"export_{stream_id}.{format if format != 'markdown' else 'md'}"
            with open(filename, 'w') as f:
                f.write(export)
            print(f"✅ Exported to {filename}")
        
        elif choice == "9":
            print("Goodbye!")
            break
        
        else:
            print("Invalid option")

async def quick_access_example():
    """Example of quick programmatic access"""
    retriever = AIOSAnalysisRetriever()
    
    print("\n=== Quick Access Examples ===\n")
    
    # 1. Get the latest recommendations
    print("1. Loading latest recommendations:")
    recommendations = retriever.load_saved_analysis("recursive_improvement_recommendations.json")
    if "recommendations" in recommendations:
        print(f"   Found {len(recommendations['agents_involved'])} agent recommendations")
        print(f"   Tokens: {recommendations['tokens_generated']}")
    
    # 2. Search for specific analyses
    print("\n2. Searching for 'security' analyses:")
    security_analyses = await retriever.search_by_keywords(["security", "safety"])
    print(f"   Found {len(security_analyses)} security-related analyses")
    
    # 3. Get analyses by specific agent
    print("\n3. Getting ChiefArchitect's analyses:")
    chief_analyses = await retriever.search_by_agent("ChiefArchitect")
    print(f"   Found {len(chief_analyses)} analyses")
    
    # 4. Export a stream
    if security_analyses:
        stream_id = security_analyses[0]["stream_id"]
        print(f"\n4. Exporting stream {stream_id} to markdown:")
        markdown = await retriever.export_analysis(stream_id, "markdown")
        print(f"   Exported {len(markdown)} characters")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Quick access example
        asyncio.run(quick_access_example())
    else:
        # Interactive mode
        asyncio.run(interactive_query())