#!/usr/bin/env python3
"""
Advanced resource monitoring and alerting script.

This script provides:
- Real-time resource monitoring
- Intelligent alerting based on patterns
- Resource usage prediction
- Auto-scaling recommendations
- Performance bottleneck detection
"""

import asyncio
import argparse
import json
import time
import psutil
import docker
import subprocess
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque

import numpy as np
import requests


@dataclass
class ResourceSnapshot:
    """Resource usage snapshot."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_io_mbps: float
    gpu_utilization: float = 0.0
    gpu_memory_percent: float = 0.0
    active_processes: int = 0
    load_average: float = 0.0


@dataclass
class Alert:
    """Resource alert."""
    alert_type: str
    severity: str  # info, warning, critical
    message: str
    metric_value: float
    threshold: float
    timestamp: datetime
    resolved: bool = False


class ResourceMonitor:
    """
    Advanced resource monitoring system.

    Features:
    - Multi-dimensional resource tracking
    - Intelligent pattern-based alerting
    - Performance prediction
    - Container resource monitoring
    - Auto-scaling recommendations
    """

    def __init__(self):
        """Initialize resource monitor."""
        self.snapshots = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        self.alerts = deque(maxlen=1000)
        self.docker_client = None

        # Alert thresholds
        self.thresholds = {
            'cpu_warning': 80.0,
            'cpu_critical': 95.0,
            'memory_warning': 85.0,
            'memory_critical': 95.0,
            'disk_warning': 80.0,
            'disk_critical': 90.0,
            'gpu_warning': 90.0,
            'gpu_critical': 98.0,
            'load_warning': psutil.cpu_count() * 2,
            'load_critical': psutil.cpu_count() * 4
        }

        # Initialize Docker client if available
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            print(f"Docker client not available: {e}")

    def collect_system_metrics(self) -> ResourceSnapshot:
        """Collect current system resource metrics."""

        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        load_avg = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0

        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent

        # Network metrics
        net_io = psutil.net_io_counters()
        # Calculate network rate (simplified)
        network_mbps = 0.0
        if len(self.snapshots) > 0:
            prev_snapshot = self.snapshots[-1]
            time_delta = (datetime.now() - prev_snapshot.timestamp).total_seconds()
            if time_delta > 0:
                # This is a simplified calculation
                network_mbps = 0.0  # Would calculate from bytes_sent/recv delta

        # GPU metrics (if available)
        gpu_util, gpu_memory = self._get_gpu_metrics()

        # Process count
        active_processes = len(psutil.pids())

        snapshot = ResourceSnapshot(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_usage_percent=disk_percent,
            network_io_mbps=network_mbps,
            gpu_utilization=gpu_util,
            gpu_memory_percent=gpu_memory,
            active_processes=active_processes,
            load_average=load_avg
        )

        self.snapshots.append(snapshot)
        return snapshot

    def _get_gpu_metrics(self) -> tuple[float, float]:
        """Get GPU utilization metrics."""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # First GPU
                return gpu.load * 100, gpu.memoryUtil * 100
        except ImportError:
            pass
        except Exception as e:
            print(f"Error getting GPU metrics: {e}")

        return 0.0, 0.0

    def collect_container_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Collect Docker container resource metrics."""

        if not self.docker_client:
            return {}

        container_metrics = {}

        try:
            containers = self.docker_client.containers.list()

            for container in containers:
                try:
                    stats = container.stats(stream=False, decode=True)

                    # Calculate CPU percentage
                    cpu_percent = 0.0
                    if 'cpu_stats' in stats and 'precpu_stats' in stats:
                        cpu_stats = stats['cpu_stats']
                        precpu_stats = stats['precpu_stats']

                        cpu_delta = cpu_stats['cpu_usage']['total_usage'] - precpu_stats['cpu_usage']['total_usage']
                        system_delta = cpu_stats['system_cpu_usage'] - precpu_stats['system_cpu_usage']

                        if system_delta > 0:
                            cpu_count = cpu_stats.get('online_cpus', len(cpu_stats['cpu_usage'].get('percpu_usage', [1])))
                            cpu_percent = (cpu_delta / system_delta) * cpu_count * 100.0

                    # Calculate memory percentage
                    memory_percent = 0.0
                    if 'memory_stats' in stats:
                        memory_usage = stats['memory_stats'].get('usage', 0)
                        memory_limit = stats['memory_stats'].get('limit', 1)
                        memory_percent = (memory_usage / memory_limit) * 100.0

                    container_metrics[container.name] = {
                        'cpu_percent': cpu_percent,
                        'memory_percent': memory_percent,
                        'memory_usage_mb': memory_usage / (1024 * 1024) if 'memory_usage' in locals() else 0,
                        'status': container.status,
                        'image': container.image.tags[0] if container.image.tags else 'unknown'
                    }

                except Exception as e:
                    container_metrics[container.name] = {'error': str(e)}

        except Exception as e:
            print(f"Error collecting container metrics: {e}")

        return container_metrics

    def check_alerts(self, snapshot: ResourceSnapshot) -> List[Alert]:
        """Check for alert conditions and generate alerts."""

        new_alerts = []

        # CPU alerts
        if snapshot.cpu_percent >= self.thresholds['cpu_critical']:
            new_alerts.append(Alert(
                alert_type="cpu_critical",
                severity="critical",
                message=f"Critical CPU usage: {snapshot.cpu_percent:.1f}%",
                metric_value=snapshot.cpu_percent,
                threshold=self.thresholds['cpu_critical'],
                timestamp=snapshot.timestamp
            ))
        elif snapshot.cpu_percent >= self.thresholds['cpu_warning']:
            new_alerts.append(Alert(
                alert_type="cpu_warning",
                severity="warning",
                message=f"High CPU usage: {snapshot.cpu_percent:.1f}%",
                metric_value=snapshot.cpu_percent,
                threshold=self.thresholds['cpu_warning'],
                timestamp=snapshot.timestamp
            ))

        # Memory alerts
        if snapshot.memory_percent >= self.thresholds['memory_critical']:
            new_alerts.append(Alert(
                alert_type="memory_critical",
                severity="critical",
                message=f"Critical memory usage: {snapshot.memory_percent:.1f}%",
                metric_value=snapshot.memory_percent,
                threshold=self.thresholds['memory_critical'],
                timestamp=snapshot.timestamp
            ))
        elif snapshot.memory_percent >= self.thresholds['memory_warning']:
            new_alerts.append(Alert(
                alert_type="memory_warning",
                severity="warning",
                message=f"High memory usage: {snapshot.memory_percent:.1f}%",
                metric_value=snapshot.memory_percent,
                threshold=self.thresholds['memory_warning'],
                timestamp=snapshot.timestamp
            ))

        # GPU alerts
        if snapshot.gpu_utilization >= self.thresholds['gpu_critical']:
            new_alerts.append(Alert(
                alert_type="gpu_critical",
                severity="critical",
                message=f"Critical GPU usage: {snapshot.gpu_utilization:.1f}%",
                metric_value=snapshot.gpu_utilization,
                threshold=self.thresholds['gpu_critical'],
                timestamp=snapshot.timestamp
            ))

        # Load average alerts
        if snapshot.load_average >= self.thresholds['load_critical']:
            new_alerts.append(Alert(
                alert_type="load_critical",
                severity="critical",
                message=f"Critical system load: {snapshot.load_average:.2f}",
                metric_value=snapshot.load_average,
                threshold=self.thresholds['load_critical'],
                timestamp=snapshot.timestamp
            ))

        # Add to alert history
        self.alerts.extend(new_alerts)

        return new_alerts

    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalous resource usage patterns."""

        if len(self.snapshots) < 60:  # Need at least 1 hour of data
            return []

        anomalies = []

        # Convert snapshots to arrays for analysis
        recent_data = list(self.snapshots)[-60:]  # Last hour
        cpu_values = [s.cpu_percent for s in recent_data]
        memory_values = [s.memory_percent for s in recent_data]

        # Detect CPU spikes
        cpu_mean = np.mean(cpu_values)
        cpu_std = np.std(cpu_values)
        cpu_threshold = cpu_mean + 2 * cpu_std  # 2 standard deviations

        recent_cpu = cpu_values[-10:]  # Last 10 minutes
        if any(cpu > cpu_threshold for cpu in recent_cpu):
            anomalies.append({
                'type': 'cpu_spike',
                'severity': 'warning',
                'description': f'CPU usage spike detected: {max(recent_cpu):.1f}% (baseline: {cpu_mean:.1f}%)',
                'metric': 'cpu_percent',
                'current_value': max(recent_cpu),
                'baseline': cpu_mean
            })

        # Detect memory leaks (gradual memory increase)
        if len(memory_values) >= 30:
            # Check if memory is consistently increasing
            recent_trend = np.polyfit(range(30), memory_values[-30:], 1)[0]  # Slope
            if recent_trend > 0.5:  # Increasing by >0.5% per minute
                anomalies.append({
                    'type': 'memory_leak',
                    'severity': 'warning',
                    'description': f'Potential memory leak detected: {recent_trend:.2f}% increase per minute',
                    'metric': 'memory_percent',
                    'current_value': memory_values[-1],
                    'trend': recent_trend
                })

        # Detect resource oscillations
        cpu_oscillation = self._detect_oscillation(cpu_values)
        if cpu_oscillation:
            anomalies.append({
                'type': 'cpu_oscillation',
                'severity': 'info',
                'description': 'CPU usage oscillation detected - may indicate inefficient workload scheduling',
                'metric': 'cpu_percent',
                'pattern': 'oscillation'
            })

        return anomalies

    def _detect_oscillation(self, values: List[float], threshold: float = 20.0) -> bool:
        """Detect oscillating patterns in metric values."""

        if len(values) < 20:
            return False

        # Simple oscillation detection: count direction changes
        direction_changes = 0
        for i in range(2, len(values)):
            prev_diff = values[i-1] - values[i-2]
            curr_diff = values[i] - values[i-1]

            # Direction changed if signs are different and change is significant
            if abs(prev_diff) > 5 and abs(curr_diff) > 5:
                if (prev_diff > 0) != (curr_diff > 0):
                    direction_changes += 1

        # If more than 30% of possible changes are direction changes, it's oscillating
        oscillation_ratio = direction_changes / (len(values) - 2)
        return oscillation_ratio > 0.3

    def predict_resource_usage(self, hours_ahead: int = 1) -> Dict[str, float]:
        """Predict resource usage for next N hours."""

        if len(self.snapshots) < 60:  # Need at least 1 hour of data
            return {}

        # Get recent trend data
        recent_data = list(self.snapshots)[-60:]  # Last hour
        timestamps = [(s.timestamp - recent_data[0].timestamp).total_seconds() / 3600 for s in recent_data]  # Hours from start

        predictions = {}

        # Predict CPU usage
        cpu_values = [s.cpu_percent for s in recent_data]
        if len(set(cpu_values)) > 1:  # Has variation
            cpu_trend = np.polyfit(timestamps, cpu_values, 1)[0]  # Linear trend
            cpu_prediction = cpu_values[-1] + (cpu_trend * hours_ahead)
            predictions['cpu_percent'] = max(0, min(100, cpu_prediction))

        # Predict memory usage
        memory_values = [s.memory_percent for s in recent_data]
        if len(set(memory_values)) > 1:  # Has variation
            memory_trend = np.polyfit(timestamps, memory_values, 1)[0]
            memory_prediction = memory_values[-1] + (memory_trend * hours_ahead)
            predictions['memory_percent'] = max(0, min(100, memory_prediction))

        return predictions

    def get_scaling_recommendations(self) -> List[Dict[str, Any]]:
        """Generate auto-scaling recommendations."""

        if len(self.snapshots) < 30:  # Need at least 30 minutes of data
            return []

        recommendations = []
        recent_data = list(self.snapshots)[-30:]  # Last 30 minutes

        # Analyze CPU usage
        cpu_values = [s.cpu_percent for s in recent_data]
        avg_cpu = np.mean(cpu_values)
        max_cpu = max(cpu_values)

        if avg_cpu > 80:
            recommendations.append({
                'type': 'scale_up',
                'resource': 'cpu',
                'reason': f'High average CPU usage: {avg_cpu:.1f}%',
                'recommended_action': 'Add CPU cores or scale horizontally',
                'urgency': 'high' if max_cpu > 95 else 'medium'
            })
        elif avg_cpu < 30 and max_cpu < 50:
            recommendations.append({
                'type': 'scale_down',
                'resource': 'cpu',
                'reason': f'Low CPU utilization: {avg_cpu:.1f}%',
                'recommended_action': 'Reduce CPU allocation or consolidate workloads',
                'urgency': 'low'
            })

        # Analyze memory usage
        memory_values = [s.memory_percent for s in recent_data]
        avg_memory = np.mean(memory_values)

        if avg_memory > 85:
            recommendations.append({
                'type': 'scale_up',
                'resource': 'memory',
                'reason': f'High memory usage: {avg_memory:.1f}%',
                'recommended_action': 'Increase memory allocation',
                'urgency': 'high'
            })

        # Analyze GPU usage (if available)
        gpu_values = [s.gpu_utilization for s in recent_data if s.gpu_utilization > 0]
        if gpu_values:
            avg_gpu = np.mean(gpu_values)
            if avg_gpu > 90:
                recommendations.append({
                    'type': 'scale_up',
                    'resource': 'gpu',
                    'reason': f'High GPU utilization: {avg_gpu:.1f}%',
                    'recommended_action': 'Add GPU resources or optimize GPU usage',
                    'urgency': 'high'
                })

        return recommendations

    async def monitor_continuously(self, interval_seconds: int = 60, duration_hours: int = 24):
        """Run continuous monitoring for specified duration."""

        print(f"🔍 Starting continuous monitoring for {duration_hours} hours...")
        print(f"📊 Collecting metrics every {interval_seconds} seconds")

        end_time = datetime.now() + timedelta(hours=duration_hours)

        while datetime.now() < end_time:
            try:
                # Collect metrics
                snapshot = self.collect_system_metrics()
                container_metrics = self.collect_container_metrics()

                # Check for alerts
                alerts = self.check_alerts(snapshot)

                # Print alerts if any
                for alert in alerts:
                    severity_emoji = "🚨" if alert.severity == "critical" else "⚠️"
                    print(f"{severity_emoji} {alert.message}")

                # Detect anomalies periodically
                if len(self.snapshots) % 10 == 0:  # Every 10 minutes
                    anomalies = self.detect_anomalies()
                    for anomaly in anomalies:
                        print(f"🔍 Anomaly detected: {anomaly['description']}")

                # Print periodic status
                if len(self.snapshots) % 15 == 0:  # Every 15 minutes
                    print(f"📈 Status - CPU: {snapshot.cpu_percent:.1f}%, "
                          f"Memory: {snapshot.memory_percent:.1f}%, "
                          f"GPU: {snapshot.gpu_utilization:.1f}%")

                await asyncio.sleep(interval_seconds)

            except KeyboardInterrupt:
                print("\n⏹️ Monitoring stopped by user")
                break
            except Exception as e:
                print(f"❌ Error during monitoring: {e}")
                await asyncio.sleep(interval_seconds)

        print("✅ Monitoring completed")

    def generate_report(self, output_file: str = None) -> Dict[str, Any]:
        """Generate comprehensive monitoring report."""

        if not self.snapshots:
            return {"error": "No monitoring data available"}

        # Calculate statistics
        recent_data = list(self.snapshots)[-60:] if len(self.snapshots) >= 60 else list(self.snapshots)

        cpu_values = [s.cpu_percent for s in recent_data]
        memory_values = [s.memory_percent for s in recent_data]
        gpu_values = [s.gpu_utilization for s in recent_data if s.gpu_utilization > 0]

        report = {
            "timestamp": datetime.now().isoformat(),
            "monitoring_period": {
                "start": recent_data[0].timestamp.isoformat(),
                "end": recent_data[-1].timestamp.isoformat(),
                "duration_hours": len(recent_data) / 60.0
            },
            "resource_statistics": {
                "cpu": {
                    "average": np.mean(cpu_values),
                    "maximum": max(cpu_values),
                    "minimum": min(cpu_values),
                    "std_deviation": np.std(cpu_values)
                },
                "memory": {
                    "average": np.mean(memory_values),
                    "maximum": max(memory_values),
                    "minimum": min(memory_values),
                    "std_deviation": np.std(memory_values)
                }
            },
            "alerts": [
                {
                    "type": alert.alert_type,
                    "severity": alert.severity,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "metric_value": alert.metric_value
                }
                for alert in list(self.alerts)[-50:]  # Last 50 alerts
            ],
            "anomalies": self.detect_anomalies(),
            "predictions": self.predict_resource_usage(hours_ahead=2),
            "scaling_recommendations": self.get_scaling_recommendations(),
            "summary": {
                "total_alerts": len(self.alerts),
                "critical_alerts": len([a for a in self.alerts if a.severity == "critical"]),
                "average_cpu_usage": np.mean(cpu_values),
                "average_memory_usage": np.mean(memory_values),
                "peak_usage_detected": max(cpu_values) > 90 or max(memory_values) > 90
            }
        }

        if gpu_values:
            report["resource_statistics"]["gpu"] = {
                "average": np.mean(gpu_values),
                "maximum": max(gpu_values),
                "minimum": min(gpu_values)
            }

        # Print summary
        print(f"\n📋 Monitoring Report Summary:")
        print(f"Monitoring Duration: {report['monitoring_period']['duration_hours']:.1f} hours")
        print(f"Total Alerts: {report['summary']['total_alerts']}")
        print(f"Critical Alerts: {report['summary']['critical_alerts']}")
        print(f"Average CPU: {report['summary']['average_cpu_usage']:.1f}%")
        print(f"Average Memory: {report['summary']['average_memory_usage']:.1f}%")

        if len(report["scaling_recommendations"]) > 0:
            print(f"Scaling Recommendations: {len(report['scaling_recommendations'])}")

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\n💾 Report saved to: {output_file}")

        return report


def main():
    """Main CLI interface."""

    parser = argparse.ArgumentParser(description="Advanced Resource Monitor")

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Start continuous monitoring')
    monitor_parser.add_argument('--interval', type=int, default=60, help='Monitoring interval in seconds')
    monitor_parser.add_argument('--duration', type=int, default=24, help='Monitoring duration in hours')

    # Snapshot command
    snapshot_parser = subparsers.add_parser('snapshot', help='Take a single resource snapshot')

    # Report command
    report_parser = subparsers.add_parser('report', help='Generate monitoring report')
    report_parser.add_argument('--output', help='Output file for report')

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict resource usage')
    predict_parser.add_parameter('--hours', type=int, default=2, help='Hours ahead to predict')

    # Containers command
    containers_parser = subparsers.add_parser('containers', help='Show container resource usage')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    monitor = ResourceMonitor()

    try:
        if args.command == 'monitor':
            asyncio.run(monitor.monitor_continuously(args.interval, args.duration))

        elif args.command == 'snapshot':
            snapshot = monitor.collect_system_metrics()
            print(f"📊 System Resource Snapshot:")
            print(f"CPU: {snapshot.cpu_percent:.1f}%")
            print(f"Memory: {snapshot.memory_percent:.1f}%")
            print(f"Disk: {snapshot.disk_usage_percent:.1f}%")
            print(f"Load Average: {snapshot.load_average:.2f}")
            if snapshot.gpu_utilization > 0:
                print(f"GPU: {snapshot.gpu_utilization:.1f}%")
                print(f"GPU Memory: {snapshot.gpu_memory_percent:.1f}%")

        elif args.command == 'report':
            # Collect some data first
            print("Collecting monitoring data...")
            for _ in range(5):  # Collect 5 snapshots
                monitor.collect_system_metrics()
                time.sleep(2)
            monitor.generate_report(args.output)

        elif args.command == 'predict':
            # Need some historical data first
            print("Collecting baseline data for prediction...")
            for _ in range(10):
                monitor.collect_system_metrics()
                time.sleep(6)  # 1 minute of data

            predictions = monitor.predict_resource_usage(args.hours)
            if predictions:
                print(f"🔮 Resource Predictions ({args.hours} hours ahead):")
                for metric, value in predictions.items():
                    print(f"{metric}: {value:.1f}%")
            else:
                print("Insufficient data for prediction")

        elif args.command == 'containers':
            container_metrics = monitor.collect_container_metrics()
            if container_metrics:
                print("🐳 Container Resource Usage:")
                for name, metrics in container_metrics.items():
                    if 'error' in metrics:
                        print(f"{name}: Error - {metrics['error']}")
                    else:
                        print(f"{name}: CPU {metrics['cpu_percent']:.1f}%, "
                              f"Memory {metrics['memory_percent']:.1f}%")
            else:
                print("No container metrics available")

    except KeyboardInterrupt:
        print("\n⏹️ Monitoring stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()