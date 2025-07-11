"""Evaluation reporting utilities for episode evaluation results."""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from ..multi_agent.state import StepEvaluationResult

logger = logging.getLogger(__name__)


class EvaluationReporter:
    """Generate comprehensive evaluation reports."""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir or "evaluation_reports")
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_episode_report(
        self, 
        episode_result: Dict[str, Any],
        save_to_file: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive episode evaluation report.
        
        Args:
            episode_result: Episode evaluation result from EpisodeRunner
            save_to_file: Whether to save report to file
            
        Returns:
            Formatted evaluation report
        """
        try:
            report = {
                "report_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "episode_id": episode_result.get("episode_id", "unknown"),
                    "evaluation_framework": "mmllm-aitw-evaluation"
                },
                "episode_summary": {
                    "episode_id": episode_result.get("episode_id"),
                    "total_steps": episode_result.get("total_steps", 0),
                    "overall_success_rate": episode_result.get("overall_success_rate", 0.0),
                    "completed_steps": self._count_successful_steps(episode_result.get("step_evaluations", [])),
                    "failed_steps": self._count_failed_steps(episode_result.get("step_evaluations", []))
                },
                "performance_metrics": self._calculate_performance_metrics(episode_result),
                "step_by_step_analysis": self._generate_step_analysis(episode_result.get("step_evaluations", [])),
                "error_analysis": self._analyze_errors(episode_result.get("step_evaluations", [])),
                "recommendations": self._generate_recommendations(episode_result)
            }
            
            if save_to_file:
                self._save_report_to_file(report, "episode")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating episode report: {e}")
            return {"error": str(e)}
    
    def generate_batch_report(
        self, 
        episode_results: List[Dict[str, Any]],
        save_to_file: bool = True
    ) -> Dict[str, Any]:
        """
        Generate batch evaluation report across multiple episodes.
        
        Args:
            episode_results: List of episode evaluation results
            save_to_file: Whether to save report to file
            
        Returns:
            Batch evaluation report
        """
        try:
            if not episode_results:
                return {"error": "No episode results provided"}
            
            report = {
                "report_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "total_episodes": len(episode_results),
                    "evaluation_framework": "mmllm-aitw-evaluation"
                },
                "aggregate_metrics": self._calculate_aggregate_metrics(episode_results),
                "performance_distribution": self._analyze_performance_distribution(episode_results),
                "action_type_analysis": self._analyze_action_types(episode_results),
                "failure_patterns": self._analyze_failure_patterns(episode_results),
                "episode_summaries": [
                    {
                        "episode_id": result.get("episode_id"),
                        "success_rate": result.get("overall_success_rate", 0.0),
                        "total_steps": result.get("total_steps", 0)
                    }
                    for result in episode_results
                ]
            }
            
            if save_to_file:
                self._save_report_to_file(report, "batch")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating batch report: {e}")
            return {"error": str(e)}
    
    def generate_markdown_report(self, report: Dict[str, Any]) -> str:
        """Generate markdown format report."""
        try:
            markdown = []
            
            # Title
            if "episode_summary" in report:
                episode_id = report["episode_summary"].get("episode_id", "Unknown")
                markdown.append(f"# Episode Evaluation Report: {episode_id}")
            else:
                markdown.append("# Batch Evaluation Report")
            
            markdown.append(f"*Generated: {report.get('report_metadata', {}).get('timestamp', 'unknown')}*")
            markdown.append("")
            
            # Summary section
            if "episode_summary" in report:
                summary = report["episode_summary"]
                markdown.extend([
                    "## Episode Summary",
                    f"- **Episode ID**: {summary.get('episode_id')}",
                    f"- **Total Steps**: {summary.get('total_steps')}",
                    f"- **Success Rate**: {summary.get('overall_success_rate', 0):.1%}",
                    f"- **Completed Steps**: {summary.get('completed_steps')}",
                    f"- **Failed Steps**: {summary.get('failed_steps')}",
                    ""
                ])
            
            elif "aggregate_metrics" in report:
                metrics = report["aggregate_metrics"]
                markdown.extend([
                    "## Aggregate Performance",
                    f"- **Total Episodes**: {report['report_metadata'].get('total_episodes')}",
                    f"- **Average Success Rate**: {metrics.get('average_success_rate', 0):.1%}",
                    f"- **Best Episode**: {metrics.get('best_success_rate', 0):.1%}",
                    f"- **Worst Episode**: {metrics.get('worst_success_rate', 0):.1%}",
                    ""
                ])
            
            # Performance metrics
            if "performance_metrics" in report:
                metrics = report["performance_metrics"]
                markdown.extend([
                    "## Performance Metrics",
                    f"- **Action Type Accuracy**: {metrics.get('action_type_accuracy', 0):.1%}",
                    f"- **Coordinate Accuracy**: {metrics.get('coordinate_accuracy', 0):.1%}",
                    f"- **Text Input Accuracy**: {metrics.get('text_accuracy', 0):.1%}",
                    f"- **Average Coordinate Distance**: {metrics.get('avg_coordinate_distance', 0):.3f}",
                    ""
                ])
            
            # Step analysis (for episode reports)
            if "step_by_step_analysis" in report:
                markdown.extend([
                    "## Step-by-Step Analysis",
                    "| Step | Action Type | Score | Match | Distance |",
                    "|------|-------------|-------|-------|----------|"
                ])
                
                for step in report["step_by_step_analysis"]:
                    step_num = step.get("step_number", 0)
                    action_type = step.get("action_type", "unknown")
                    score = step.get("evaluation_score", 0)
                    match = "✓" if step.get("action_match", False) else "✗"
                    distance = step.get("coordinate_distance", "N/A")
                    if distance != "N/A":
                        distance = f"{distance:.3f}"
                    
                    markdown.append(f"| {step_num} | {action_type} | {score:.2f} | {match} | {distance} |")
                
                markdown.append("")
            
            # Error analysis
            if "error_analysis" in report:
                errors = report["error_analysis"]
                markdown.extend([
                    "## Error Analysis",
                    f"- **Coordinate Errors**: {errors.get('coordinate_errors', 0)}",
                    f"- **Action Type Errors**: {errors.get('action_type_errors', 0)}",
                    f"- **Text Input Errors**: {errors.get('text_errors', 0)}",
                    ""
                ])
            
            # Recommendations
            if "recommendations" in report:
                markdown.extend([
                    "## Recommendations",
                    ""
                ])
                for rec in report["recommendations"]:
                    markdown.append(f"- {rec}")
                markdown.append("")
            
            return "\n".join(markdown)
            
        except Exception as e:
            logger.error(f"Error generating markdown report: {e}")
            return f"Error generating markdown report: {str(e)}"
    
    def _count_successful_steps(self, step_evaluations: List[StepEvaluationResult]) -> int:
        """Count number of successful steps."""
        return sum(1 for step in step_evaluations if step.action_match)
    
    def _count_failed_steps(self, step_evaluations: List[StepEvaluationResult]) -> int:
        """Count number of failed steps."""
        return sum(1 for step in step_evaluations if not step.action_match)
    
    def _calculate_performance_metrics(self, episode_result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate detailed performance metrics."""
        step_evaluations = episode_result.get("step_evaluations", [])
        if not step_evaluations:
            return {}
        
        total_steps = len(step_evaluations)
        action_type_matches = sum(1 for step in step_evaluations if step.action_type_match)
        coordinate_matches = sum(1 for step in step_evaluations 
                               if step.coordinate_distance is not None and step.coordinate_distance <= 0.14)
        text_matches = sum(1 for step in step_evaluations 
                          if step.text_match is not None and step.text_match)
        
        # Calculate average coordinate distance
        coord_distances = [step.coordinate_distance for step in step_evaluations 
                          if step.coordinate_distance is not None]
        avg_coord_distance = sum(coord_distances) / len(coord_distances) if coord_distances else 0
        
        return {
            "action_type_accuracy": action_type_matches / total_steps if total_steps > 0 else 0,
            "coordinate_accuracy": coordinate_matches / total_steps if total_steps > 0 else 0,
            "text_accuracy": text_matches / total_steps if total_steps > 0 else 0,
            "avg_coordinate_distance": avg_coord_distance
        }
    
    def _generate_step_analysis(self, step_evaluations: List[StepEvaluationResult]) -> List[Dict[str, Any]]:
        """Generate detailed step-by-step analysis."""
        analysis = []
        
        for step in step_evaluations:
            step_info = {
                "step_number": step.step_number,
                "action_type": step.agent_action.get("action_type", "unknown"),
                "evaluation_score": step.evaluation_score,
                "action_match": step.action_match,
                "action_type_match": step.action_type_match,
                "coordinate_distance": step.coordinate_distance,
                "text_match": step.text_match
            }
            analysis.append(step_info)
        
        return analysis
    
    def _analyze_errors(self, step_evaluations: List[StepEvaluationResult]) -> Dict[str, int]:
        """Analyze common error patterns."""
        return {
            "coordinate_errors": sum(1 for step in step_evaluations 
                                   if step.coordinate_distance is not None and step.coordinate_distance > 0.14),
            "action_type_errors": sum(1 for step in step_evaluations if not step.action_type_match),
            "text_errors": sum(1 for step in step_evaluations 
                             if step.text_match is not None and not step.text_match)
        }
    
    def _generate_recommendations(self, episode_result: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        success_rate = episode_result.get("overall_success_rate", 0.0)
        step_evaluations = episode_result.get("step_evaluations", [])
        
        if success_rate < 0.5:
            recommendations.append("Overall success rate is low. Consider improving vision processing or prompt engineering.")
        
        if step_evaluations:
            coord_errors = sum(1 for step in step_evaluations 
                             if step.coordinate_distance is not None and step.coordinate_distance > 0.14)
            if coord_errors > len(step_evaluations) * 0.3:
                recommendations.append("High coordinate error rate. Consider improving UI element detection.")
            
            type_errors = sum(1 for step in step_evaluations if not step.action_type_match)
            if type_errors > len(step_evaluations) * 0.2:
                recommendations.append("High action type error rate. Consider improving action classification.")
        
        if not recommendations:
            recommendations.append("Good performance! Consider testing on more diverse episodes.")
        
        return recommendations
    
    def _calculate_aggregate_metrics(self, episode_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate aggregate metrics across episodes."""
        if not episode_results:
            return {}
        
        success_rates = [result.get("overall_success_rate", 0.0) for result in episode_results]
        
        return {
            "average_success_rate": sum(success_rates) / len(success_rates),
            "best_success_rate": max(success_rates),
            "worst_success_rate": min(success_rates),
            "total_episodes": len(episode_results)
        }
    
    def _analyze_performance_distribution(self, episode_results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze distribution of performance levels."""
        success_rates = [result.get("overall_success_rate", 0.0) for result in episode_results]
        
        return {
            "excellent": sum(1 for rate in success_rates if rate >= 0.9),
            "good": sum(1 for rate in success_rates if 0.7 <= rate < 0.9),
            "fair": sum(1 for rate in success_rates if 0.5 <= rate < 0.7),
            "poor": sum(1 for rate in success_rates if rate < 0.5)
        }
    
    def _analyze_action_types(self, episode_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance by action type."""
        # This would need more detailed step information
        return {"analysis": "Action type analysis requires step-level data"}
    
    def _analyze_failure_patterns(self, episode_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze common failure patterns."""
        # This would need more detailed error information
        return {"analysis": "Failure pattern analysis requires step-level error data"}
    
    def _save_report_to_file(self, report: Dict[str, Any], report_type: str) -> None:
        """Save report to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if report_type == "episode":
                episode_id = report.get("episode_summary", {}).get("episode_id", "unknown")
                filename = f"episode_report_{episode_id}_{timestamp}.json"
            else:
                filename = f"batch_report_{timestamp}.json"
            
            filepath = self.output_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Also save markdown version
            markdown_content = self.generate_markdown_report(report)
            markdown_filepath = filepath.with_suffix('.md')
            with open(markdown_filepath, 'w') as f:
                f.write(markdown_content)
            
            logger.info(f"Report saved to {filepath} and {markdown_filepath}")
            
        except Exception as e:
            logger.error(f"Error saving report to file: {e}")
