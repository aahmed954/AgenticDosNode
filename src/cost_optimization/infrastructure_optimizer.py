"""
Infrastructure optimization system for the agentic AI stack.

This module provides:
- Resource allocation optimization between thanos/oracle1
- Container resource optimization
- Auto-scaling strategies
- Network and storage cost optimization
- Cross-node workload distribution
"""

from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json
from collections import defaultdict, deque
import psutil
import docker
import subprocess
import numpy as np

from ..config import settings
from ..utils.logging import get_logger
from .cost_tracker import CostTracker, CostCategory

logger = get_logger(__name__)


class NodeType(str, Enum):
    """Types of nodes in the infrastructure."""
    GPU_NODE = "gpu_node"      # thanos - RTX 4090
    CPU_NODE = "cpu_node"      # oracle1 - CPU only
    HYBRID_NODE = "hybrid_node"  # Mixed workloads


class ResourceType(str, Enum):
    """Types of resources to optimize."""
    GPU_MEMORY = "gpu_memory"
    CPU_CORES = "cpu_cores"
    SYSTEM_MEMORY = "system_memory"
    STORAGE = "storage"
    NETWORK_BANDWIDTH = "network_bandwidth"


class WorkloadType(str, Enum):
    """Types of workloads."""
    LLM_INFERENCE = "llm_inference"
    EMBEDDING_GENERATION = "embedding_generation"
    VECTOR_SEARCH = "vector_search"
    DATABASE_OPERATIONS = "database_operations"
    CACHING = "caching"
    MONITORING = "monitoring"
    API_GATEWAY = "api_gateway"
    BACKGROUND_TASKS = "background_tasks"


@dataclass
class NodeResources:
    """Resource specifications for a node."""
    node_id: str
    node_type: NodeType
    gpu_count: int = 0
    gpu_memory_gb: int = 0
    cpu_cores: int = 0
    cpu_threads: int = 0
    system_memory_gb: int = 0
    storage_gb: int = 0
    network_bandwidth_gbps: float = 0.0
    hourly_cost: float = 0.0


@dataclass
class ResourceUsage:
    """Current resource usage."""
    timestamp: datetime
    node_id: str
    cpu_percent: float
    memory_percent: float
    gpu_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    disk_io_mbps: float = 0.0
    network_io_mbps: float = 0.0


@dataclass
class WorkloadProfile:
    """Profile of a workload's resource requirements."""
    workload_type: WorkloadType
    avg_cpu_cores: float
    avg_memory_gb: float
    avg_gpu_memory_gb: float = 0.0
    avg_duration_seconds: float = 0.0
    requests_per_hour: int = 0
    peak_multiplier: float = 2.0  # Peak usage vs average
    can_batch: bool = False
    can_queue: bool = True
    priority: int = 1  # 1-5, 5 is highest


@dataclass
class OptimizationRecommendation:
    """Infrastructure optimization recommendation."""
    type: str
    priority: str  # high, medium, low
    title: str
    description: str
    estimated_savings: float
    implementation_effort: str  # low, medium, high
    actions: List[str]
    metrics_to_track: List[str]


class InfrastructureOptimizer:
    """
    Comprehensive infrastructure optimization system.

    Features:
    - Dynamic workload distribution between nodes
    - Resource allocation optimization
    - Auto-scaling based on demand patterns
    - Cost-aware container scheduling
    - Network and storage optimization
    - Performance monitoring and alerting
    """

    def __init__(self, cost_tracker: Optional[CostTracker] = None):
        """Initialize infrastructure optimizer."""
        self.cost_tracker = cost_tracker

        # Node configurations
        self.nodes = self._initialize_node_configs()
        self.workload_profiles = self._initialize_workload_profiles()

        # Current state tracking
        self.current_usage: Dict[str, ResourceUsage] = {}
        self.workload_history: deque = deque(maxlen=10000)
        self.optimization_history: deque = deque(maxlen=1000)

        # Docker clients for each node
        self.docker_clients = {}
        self._initialize_docker_clients()

        # Performance monitoring
        self.resource_monitoring_interval = 30  # seconds
        self.monitoring_task = None

        logger.info("Infrastructure optimizer initialized")

    def _initialize_node_configs(self) -> Dict[str, NodeResources]:
        """Initialize node resource configurations."""
        return {
            "thanos": NodeResources(
                node_id="thanos",
                node_type=NodeType.GPU_NODE,
                gpu_count=1,
                gpu_memory_gb=24,  # RTX 4090
                cpu_cores=12,
                cpu_threads=24,
                system_memory_gb=64,
                storage_gb=2000,
                network_bandwidth_gbps=1.0,
                hourly_cost=0.50  # Estimated GPU compute cost
            ),
            "oracle1": NodeResources(
                node_id="oracle1",
                node_type=NodeType.CPU_NODE,
                gpu_count=0,
                gpu_memory_gb=0,
                cpu_cores=8,
                cpu_threads=16,
                system_memory_gb=32,
                storage_gb=1000,
                network_bandwidth_gbps=1.0,
                hourly_cost=0.10  # VPS cost
            )
        }

    def _initialize_workload_profiles(self) -> Dict[WorkloadType, WorkloadProfile]:
        """Initialize workload resource profiles."""
        return {
            WorkloadType.LLM_INFERENCE: WorkloadProfile(
                workload_type=WorkloadType.LLM_INFERENCE,
                avg_cpu_cores=0.5,
                avg_memory_gb=2.0,
                avg_gpu_memory_gb=8.0,
                avg_duration_seconds=5.0,
                requests_per_hour=100,
                peak_multiplier=3.0,
                can_batch=True,
                can_queue=True,
                priority=4
            ),
            WorkloadType.EMBEDDING_GENERATION: WorkloadProfile(
                workload_type=WorkloadType.EMBEDDING_GENERATION,
                avg_cpu_cores=0.2,
                avg_memory_gb=1.0,
                avg_gpu_memory_gb=2.0,
                avg_duration_seconds=1.0,
                requests_per_hour=500,
                peak_multiplier=2.0,
                can_batch=True,
                can_queue=True,
                priority=3
            ),
            WorkloadType.VECTOR_SEARCH: WorkloadProfile(
                workload_type=WorkloadType.VECTOR_SEARCH,
                avg_cpu_cores=1.0,
                avg_memory_gb=4.0,
                avg_gpu_memory_gb=0.0,
                avg_duration_seconds=0.5,
                requests_per_hour=1000,
                peak_multiplier=5.0,
                can_batch=False,
                can_queue=True,
                priority=4
            ),
            WorkloadType.DATABASE_OPERATIONS: WorkloadProfile(
                workload_type=WorkloadType.DATABASE_OPERATIONS,
                avg_cpu_cores=0.5,
                avg_memory_gb=2.0,
                avg_gpu_memory_gb=0.0,
                avg_duration_seconds=0.1,
                requests_per_hour=2000,
                peak_multiplier=4.0,
                can_batch=True,
                can_queue=True,
                priority=3
            ),
            WorkloadType.CACHING: WorkloadProfile(
                workload_type=WorkloadType.CACHING,
                avg_cpu_cores=0.1,
                avg_memory_gb=8.0,
                avg_gpu_memory_gb=0.0,
                avg_duration_seconds=0.01,
                requests_per_hour=5000,
                peak_multiplier=2.0,
                can_batch=False,
                can_queue=False,
                priority=5
            ),
            WorkloadType.MONITORING: WorkloadProfile(
                workload_type=WorkloadType.MONITORING,
                avg_cpu_cores=0.2,
                avg_memory_gb=1.0,
                avg_gpu_memory_gb=0.0,
                avg_duration_seconds=1.0,
                requests_per_hour=3600,  # Continuous
                peak_multiplier=1.2,
                can_batch=True,
                can_queue=False,
                priority=2
            ),
            WorkloadType.API_GATEWAY: WorkloadProfile(
                workload_type=WorkloadType.API_GATEWAY,
                avg_cpu_cores=0.3,
                avg_memory_gb=0.5,
                avg_gpu_memory_gb=0.0,
                avg_duration_seconds=0.05,
                requests_per_hour=10000,
                peak_multiplier=6.0,
                can_batch=False,
                can_queue=False,
                priority=5
            ),
        }

    def _initialize_docker_clients(self):
        """Initialize Docker clients for each node."""
        try:
            # Local client for current node
            self.docker_clients["local"] = docker.from_env()

            # For remote nodes, you'd configure remote Docker clients
            # self.docker_clients["thanos"] = docker.DockerClient(base_url="tcp://thanos:2376")
            # self.docker_clients["oracle1"] = docker.DockerClient(base_url="tcp://oracle1:2376")

        except Exception as e:
            logger.warning(f"Failed to initialize Docker clients: {e}")

    async def start_monitoring(self):
        """Start continuous resource monitoring."""
        if self.monitoring_task is None:
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Started infrastructure monitoring")

    async def stop_monitoring(self):
        """Stop resource monitoring."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
            logger.info("Stopped infrastructure monitoring")

    async def _monitoring_loop(self):
        """Continuous monitoring loop."""
        while True:
            try:
                await self._collect_resource_metrics()
                await asyncio.sleep(self.resource_monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    async def _collect_resource_metrics(self):
        """Collect current resource usage metrics."""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk_io = psutil.disk_io_counters()
            network_io = psutil.net_io_counters()

            # Get GPU metrics if available
            gpu_percent = 0.0
            gpu_memory_percent = 0.0

            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Assuming single GPU
                    gpu_percent = gpu.load * 100
                    gpu_memory_percent = gpu.memoryUtil * 100
            except ImportError:
                pass  # GPU monitoring not available

            usage = ResourceUsage(
                timestamp=datetime.utcnow(),
                node_id="local",  # Would be dynamic in multi-node setup
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                gpu_percent=gpu_percent,
                gpu_memory_percent=gpu_memory_percent,
                disk_io_mbps=0.0,  # Would calculate from counters
                network_io_mbps=0.0  # Would calculate from counters
            )

            self.current_usage["local"] = usage

            # Log to cost tracker if available
            if self.cost_tracker:
                # Calculate infrastructure costs
                node = self.nodes.get("local")
                if node:
                    usage_hours = self.resource_monitoring_interval / 3600
                    await self.cost_tracker.record_infrastructure_usage(
                        resource_type="cpu",
                        usage_hours=usage_hours,
                        metadata={
                            "cpu_percent": cpu_percent,
                            "memory_percent": memory.percent,
                            "gpu_percent": gpu_percent
                        }
                    )

        except Exception as e:
            logger.error(f"Failed to collect resource metrics: {e}")

    def get_optimal_node_for_workload(
        self,
        workload_type: WorkloadType,
        estimated_duration: Optional[float] = None,
        priority: int = 1
    ) -> str:
        """Determine optimal node for a workload."""

        if workload_type not in self.workload_profiles:
            logger.warning(f"Unknown workload type: {workload_type}")
            return "oracle1"  # Default to CPU node

        profile = self.workload_profiles[workload_type]

        # GPU workloads must go to GPU node
        if profile.avg_gpu_memory_gb > 0:
            return "thanos"

        # High-performance workloads prefer GPU node for CPU power
        if profile.priority >= 4 and profile.avg_cpu_cores > 2:
            return "thanos"

        # Check current resource availability
        for node_id, node in self.nodes.items():
            if node_id in self.current_usage:
                usage = self.current_usage[node_id]

                # Check if node has capacity
                cpu_available = node.cpu_cores * (1 - usage.cpu_percent / 100)
                memory_available = node.system_memory_gb * (1 - usage.memory_percent / 100)

                if (cpu_available >= profile.avg_cpu_cores and
                    memory_available >= profile.avg_memory_gb):
                    return node_id

        # Default assignment based on workload characteristics
        if workload_type in [WorkloadType.DATABASE_OPERATIONS,
                           WorkloadType.API_GATEWAY,
                           WorkloadType.CACHING]:
            return "oracle1"  # CPU-intensive but not GPU-requiring
        else:
            return "thanos"  # Default to more powerful node

    async def optimize_container_resources(
        self,
        workload_type: WorkloadType,
        current_limits: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize container resource limits."""

        if workload_type not in self.workload_profiles:
            return current_limits

        profile = self.workload_profiles[workload_type]

        # Calculate optimal resource allocations
        optimal_limits = {
            "cpu": f"{profile.avg_cpu_cores * profile.peak_multiplier}",
            "memory": f"{int(profile.avg_memory_gb * profile.peak_multiplier * 1024)}M",
        }

        # Add GPU resources if needed
        if profile.avg_gpu_memory_gb > 0:
            optimal_limits["device_requests"] = [
                docker.types.DeviceRequest(
                    count=1,
                    capabilities=[["gpu"]],
                    device_ids=["0"]
                )
            ]

        # Apply safety margins
        safety_margin = 1.2
        if "cpu" in optimal_limits:
            cpu_value = float(optimal_limits["cpu"])
            optimal_limits["cpu"] = str(cpu_value * safety_margin)

        if "memory" in optimal_limits:
            memory_mb = int(optimal_limits["memory"].rstrip("M"))
            optimal_limits["memory"] = f"{int(memory_mb * safety_margin)}M"

        return optimal_limits

    def calculate_scaling_requirements(
        self,
        workload_type: WorkloadType,
        current_load: float,  # Requests per second
        target_latency_ms: float = 1000.0
    ) -> Dict[str, Any]:
        """Calculate auto-scaling requirements."""

        if workload_type not in self.workload_profiles:
            return {"replicas": 1, "reason": "unknown workload"}

        profile = self.workload_profiles[workload_type]

        # Convert to requests per second
        current_rps = current_load
        profile_rps = profile.requests_per_hour / 3600

        # Calculate required capacity
        if profile_rps == 0:
            capacity_ratio = 1.0
        else:
            capacity_ratio = current_rps / profile_rps

        # Account for target latency
        latency_factor = max(1.0, profile.avg_duration_seconds * 1000 / target_latency_ms)

        # Calculate required replicas
        base_replicas = max(1, int(capacity_ratio * latency_factor))

        # Apply peak traffic multiplier
        peak_replicas = int(base_replicas * profile.peak_multiplier)

        # Apply workload-specific constraints
        if not profile.can_queue:
            # Services that can't queue need more replicas
            peak_replicas = int(peak_replicas * 1.5)

        if profile.can_batch:
            # Batch-capable services need fewer replicas
            peak_replicas = max(1, int(peak_replicas * 0.7))

        return {
            "replicas": peak_replicas,
            "reason": f"Load factor: {capacity_ratio:.2f}, Latency factor: {latency_factor:.2f}",
            "resource_requirements": {
                "cpu": profile.avg_cpu_cores * peak_replicas,
                "memory_gb": profile.avg_memory_gb * peak_replicas,
                "gpu_memory_gb": profile.avg_gpu_memory_gb * peak_replicas
            }
        }

    async def optimize_workload_distribution(
        self,
        workloads: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Optimize workload distribution across nodes."""

        distribution = {node_id: [] for node_id in self.nodes.keys()}

        # Sort workloads by priority and resource requirements
        sorted_workloads = sorted(
            workloads,
            key=lambda w: (
                -self.workload_profiles.get(w.get("type", WorkloadType.BACKGROUND_TASKS),
                                          WorkloadProfile(WorkloadType.BACKGROUND_TASKS, 0, 0)).priority,
                -self.workload_profiles.get(w.get("type", WorkloadType.BACKGROUND_TASKS),
                                          WorkloadProfile(WorkloadType.BACKGROUND_TASKS, 0, 0)).avg_gpu_memory_gb
            )
        )

        # Track available resources
        available_resources = {}
        for node_id, node in self.nodes.items():
            available_resources[node_id] = {
                "cpu_cores": node.cpu_cores,
                "memory_gb": node.system_memory_gb,
                "gpu_memory_gb": node.gpu_memory_gb
            }

            # Account for current usage
            if node_id in self.current_usage:
                usage = self.current_usage[node_id]
                available_resources[node_id]["cpu_cores"] *= (1 - usage.cpu_percent / 100)
                available_resources[node_id]["memory_gb"] *= (1 - usage.memory_percent / 100)
                if usage.gpu_memory_percent > 0:
                    available_resources[node_id]["gpu_memory_gb"] *= (1 - usage.gpu_memory_percent / 100)

        # Assign workloads to nodes
        for workload in sorted_workloads:
            workload_type = workload.get("type", WorkloadType.BACKGROUND_TASKS)
            profile = self.workload_profiles.get(workload_type)

            if not profile:
                # Assign to least loaded node
                least_loaded_node = min(
                    self.nodes.keys(),
                    key=lambda n: len(distribution[n])
                )
                distribution[least_loaded_node].append(workload)
                continue

            # Find best node for this workload
            best_node = None
            best_score = -1

            for node_id, available in available_resources.items():
                # Check if node can accommodate workload
                if (available["cpu_cores"] >= profile.avg_cpu_cores and
                    available["memory_gb"] >= profile.avg_memory_gb and
                    available["gpu_memory_gb"] >= profile.avg_gpu_memory_gb):

                    # Calculate assignment score
                    node = self.nodes[node_id]

                    # Prefer GPU nodes for GPU workloads
                    gpu_match_score = 1.0 if (profile.avg_gpu_memory_gb > 0) == (node.gpu_count > 0) else 0.5

                    # Prefer less loaded nodes
                    load_score = 1.0 - (len(distribution[node_id]) / 10.0)  # Assume max 10 workloads per node

                    # Cost efficiency
                    cost_score = 1.0 / node.hourly_cost if node.hourly_cost > 0 else 1.0

                    total_score = gpu_match_score * 0.4 + load_score * 0.4 + cost_score * 0.2

                    if total_score > best_score:
                        best_score = total_score
                        best_node = node_id

            if best_node:
                distribution[best_node].append(workload)

                # Update available resources
                available_resources[best_node]["cpu_cores"] -= profile.avg_cpu_cores
                available_resources[best_node]["memory_gb"] -= profile.avg_memory_gb
                available_resources[best_node]["gpu_memory_gb"] -= profile.avg_gpu_memory_gb
            else:
                # No suitable node found, assign to least loaded
                least_loaded_node = min(
                    self.nodes.keys(),
                    key=lambda n: len(distribution[n])
                )
                distribution[least_loaded_node].append(workload)

        return distribution

    def get_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate infrastructure optimization recommendations."""

        recommendations = []

        # Analyze current resource utilization
        for node_id, usage in self.current_usage.items():
            node = self.nodes[node_id]

            # High CPU utilization recommendation
            if usage.cpu_percent > 80:
                recommendations.append(OptimizationRecommendation(
                    type="resource_optimization",
                    priority="high",
                    title=f"High CPU utilization on {node_id}",
                    description=f"CPU usage is at {usage.cpu_percent:.1f}% on {node_id}. "
                               f"Consider scaling or load balancing.",
                    estimated_savings=node.hourly_cost * 24 * 30 * 0.2,  # 20% efficiency gain
                    implementation_effort="medium",
                    actions=[
                        f"Scale out workloads from {node_id}",
                        "Implement CPU-efficient algorithms",
                        "Add horizontal scaling rules"
                    ],
                    metrics_to_track=["cpu_utilization", "response_time", "throughput"]
                ))

            # High memory utilization recommendation
            if usage.memory_percent > 85:
                recommendations.append(OptimizationRecommendation(
                    type="resource_optimization",
                    priority="high",
                    title=f"High memory utilization on {node_id}",
                    description=f"Memory usage is at {usage.memory_percent:.1f}% on {node_id}. "
                               f"Risk of OOM errors.",
                    estimated_savings=node.hourly_cost * 24 * 30 * 0.15,
                    implementation_effort="low",
                    actions=[
                        "Implement memory limits on containers",
                        "Add swap space if needed",
                        "Optimize memory-heavy operations"
                    ],
                    metrics_to_track=["memory_utilization", "swap_usage", "container_restarts"]
                ))

            # GPU underutilization (for GPU nodes)
            if node.gpu_count > 0 and usage.gpu_percent < 30:
                recommendations.append(OptimizationRecommendation(
                    type="cost_optimization",
                    priority="medium",
                    title=f"GPU underutilization on {node_id}",
                    description=f"GPU usage is only {usage.gpu_percent:.1f}% on {node_id}. "
                               f"Consider consolidating workloads or switching to CPU for some tasks.",
                    estimated_savings=node.hourly_cost * 24 * 30 * 0.4,  # Significant GPU cost
                    implementation_effort="medium",
                    actions=[
                        "Move GPU-unnecessary workloads to CPU nodes",
                        "Implement GPU sharing strategies",
                        "Consider smaller GPU instance if available"
                    ],
                    metrics_to_track=["gpu_utilization", "gpu_memory_usage", "workload_performance"]
                ))

        # Network optimization recommendations
        total_network_cost = sum(node.hourly_cost * 0.1 for node in self.nodes.values())  # 10% for network
        if total_network_cost > 50:  # Monthly
            recommendations.append(OptimizationRecommendation(
                type="network_optimization",
                priority="low",
                title="Optimize network costs",
                description="Consider implementing network optimizations to reduce data transfer costs.",
                estimated_savings=total_network_cost * 0.3,  # 30% reduction
                implementation_effort="medium",
                actions=[
                    "Implement request/response compression",
                    "Use local caching for frequently accessed data",
                    "Optimize API payload sizes",
                    "Implement CDN for static assets"
                ],
                metrics_to_track=["network_io", "response_size", "cache_hit_rate"]
            ))

        # Storage optimization recommendations
        storage_recommendations = self._analyze_storage_optimization()
        recommendations.extend(storage_recommendations)

        # Container optimization recommendations
        container_recommendations = await self._analyze_container_optimization()
        recommendations.extend(container_recommendations)

        return sorted(recommendations, key=lambda x: x.estimated_savings, reverse=True)

    def _analyze_storage_optimization(self) -> List[OptimizationRecommendation]:
        """Analyze storage usage and generate optimization recommendations."""
        recommendations = []

        total_storage = sum(node.storage_gb for node in self.nodes.values())
        storage_cost_per_gb = 0.10  # Monthly cost per GB

        if total_storage > 1000:  # More than 1TB
            recommendations.append(OptimizationRecommendation(
                type="storage_optimization",
                priority="medium",
                title="Implement storage lifecycle policies",
                description=f"You have {total_storage}GB of storage. Implement automated cleanup "
                           f"and archiving to reduce costs.",
                estimated_savings=total_storage * storage_cost_per_gb * 0.4,  # 40% reduction
                implementation_effort="medium",
                actions=[
                    "Implement log rotation and cleanup",
                    "Archive old model checkpoints",
                    "Compress infrequently accessed data",
                    "Implement automated backup cleanup"
                ],
                metrics_to_track=["disk_usage", "storage_costs", "data_access_patterns"]
            ))

        return recommendations

    async def _analyze_container_optimization(self) -> List[OptimizationRecommendation]:
        """Analyze container resource usage and generate recommendations."""
        recommendations = []

        try:
            if "local" in self.docker_clients:
                client = self.docker_clients["local"]
                containers = client.containers.list()

                # Analyze container resource allocation vs usage
                over_provisioned = []
                under_provisioned = []

                for container in containers:
                    try:
                        stats = container.stats(stream=False)

                        # Get resource limits
                        limits = container.attrs.get("HostConfig", {})

                        # Analyze CPU usage vs limits
                        cpu_usage_percent = 0  # Would calculate from stats
                        memory_usage_percent = 0  # Would calculate from stats

                        if cpu_usage_percent < 20:  # Underutilized
                            over_provisioned.append(container.name)
                        elif cpu_usage_percent > 90:  # Over-utilized
                            under_provisioned.append(container.name)

                    except Exception as e:
                        logger.debug(f"Could not analyze container {container.name}: {e}")

                if over_provisioned:
                    recommendations.append(OptimizationRecommendation(
                        type="container_optimization",
                        priority="medium",
                        title="Reduce over-provisioned container resources",
                        description=f"Containers {', '.join(over_provisioned)} are under-utilizing "
                                   f"their allocated resources.",
                        estimated_savings=len(over_provisioned) * 10.0,  # $10 per container
                        implementation_effort="low",
                        actions=[
                            "Reduce CPU and memory limits for under-utilized containers",
                            "Implement resource monitoring and alerts",
                            "Use resource requests instead of limits where appropriate"
                        ],
                        metrics_to_track=["container_cpu_usage", "container_memory_usage", "performance_impact"]
                    ))

                if under_provisioned:
                    recommendations.append(OptimizationRecommendation(
                        type="performance_optimization",
                        priority="high",
                        title="Increase resources for over-utilized containers",
                        description=f"Containers {', '.join(under_provisioned)} are resource-constrained.",
                        estimated_savings=0.0,  # This costs money but improves performance
                        implementation_effort="low",
                        actions=[
                            "Increase CPU and memory limits for constrained containers",
                            "Consider horizontal scaling instead of vertical scaling",
                            "Implement auto-scaling policies"
                        ],
                        metrics_to_track=["container_cpu_usage", "container_memory_usage", "response_times"]
                    ))

        except Exception as e:
            logger.error(f"Failed to analyze container optimization: {e}")

        return recommendations

    async def implement_auto_scaling(
        self,
        service_name: str,
        workload_type: WorkloadType,
        current_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Implement auto-scaling decision for a service."""

        if workload_type not in self.workload_profiles:
            return {"action": "none", "reason": "unknown workload type"}

        profile = self.workload_profiles[workload_type]

        # Current performance metrics
        current_rps = current_metrics.get("requests_per_second", 0)
        current_latency = current_metrics.get("avg_latency_ms", 0)
        current_error_rate = current_metrics.get("error_rate", 0)
        current_replicas = current_metrics.get("replica_count", 1)

        # Scaling decision logic
        scale_up_conditions = [
            current_latency > 5000,  # High latency
            current_error_rate > 0.05,  # High error rate
            current_rps > profile.requests_per_hour / 3600 * profile.peak_multiplier * 0.8  # Approaching capacity
        ]

        scale_down_conditions = [
            current_latency < 1000,  # Low latency
            current_error_rate < 0.01,  # Low error rate
            current_rps < profile.requests_per_hour / 3600 * 0.3,  # Low load
            current_replicas > 1  # Has replicas to scale down
        ]

        action = "none"
        new_replica_count = current_replicas
        reason = "Metrics within normal range"

        if any(scale_up_conditions):
            # Scale up
            scaling_factor = 1.5 if current_latency > 10000 else 1.25
            new_replica_count = min(10, int(current_replicas * scaling_factor))  # Max 10 replicas
            action = "scale_up"
            reason = f"High load detected - latency: {current_latency}ms, RPS: {current_rps}"

        elif all(scale_down_conditions):
            # Scale down
            new_replica_count = max(1, int(current_replicas * 0.75))
            action = "scale_down"
            reason = f"Low load detected - can reduce resources"

        # Cost impact calculation
        node = self.get_optimal_node_for_workload(workload_type)
        node_cost = self.nodes[node].hourly_cost
        resource_cost_per_replica = (profile.avg_cpu_cores / self.nodes[node].cpu_cores) * node_cost

        cost_impact = (new_replica_count - current_replicas) * resource_cost_per_replica

        return {
            "action": action,
            "current_replicas": current_replicas,
            "new_replicas": new_replica_count,
            "reason": reason,
            "cost_impact_hourly": cost_impact,
            "estimated_performance_change": {
                "latency_improvement_pct": max(0, (current_replicas / new_replica_count - 1) * 100)
            }
        }

    def get_infrastructure_metrics(self) -> Dict[str, Any]:
        """Get comprehensive infrastructure metrics."""

        metrics = {
            "nodes": {},
            "total_hourly_cost": sum(node.hourly_cost for node in self.nodes.values()),
            "workload_distribution": {},
            "optimization_opportunities": len(self.get_optimization_recommendations()),
        }

        # Node-specific metrics
        for node_id, node in self.nodes.items():
            node_metrics = {
                "type": node.node_type.value,
                "resources": {
                    "cpu_cores": node.cpu_cores,
                    "memory_gb": node.system_memory_gb,
                    "gpu_count": node.gpu_count,
                    "gpu_memory_gb": node.gpu_memory_gb,
                    "storage_gb": node.storage_gb
                },
                "hourly_cost": node.hourly_cost
            }

            if node_id in self.current_usage:
                usage = self.current_usage[node_id]
                node_metrics["current_usage"] = {
                    "cpu_percent": usage.cpu_percent,
                    "memory_percent": usage.memory_percent,
                    "gpu_percent": usage.gpu_percent,
                    "gpu_memory_percent": usage.gpu_memory_percent
                }

                # Calculate efficiency score
                avg_utilization = (usage.cpu_percent + usage.memory_percent) / 2
                if node.gpu_count > 0:
                    avg_utilization = (avg_utilization + usage.gpu_percent) / 2

                node_metrics["efficiency_score"] = avg_utilization / 100.0

            metrics["nodes"][node_id] = node_metrics

        # Workload distribution metrics
        for workload_type, profile in self.workload_profiles.items():
            optimal_node = self.get_optimal_node_for_workload(workload_type)
            metrics["workload_distribution"][workload_type.value] = {
                "optimal_node": optimal_node,
                "resource_requirements": {
                    "cpu_cores": profile.avg_cpu_cores,
                    "memory_gb": profile.avg_memory_gb,
                    "gpu_memory_gb": profile.avg_gpu_memory_gb
                },
                "priority": profile.priority
            }

        return metrics