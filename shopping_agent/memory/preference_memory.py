"""
PreferenceMemory — 用户偏好记忆层。

融合三类记忆：
  1. 短期工作记忆（当前会话状态）
  2. 长期用户画像（历史行为积累）
  3. 隐式行为偏好（点击、停留、跳过信号）

生产环境应接入 Redis（短期）+ 数据库（长期画像）。
当前实现为内存 stub，接口与生产版本保持一致。
"""

from __future__ import annotations

import copy
from datetime import datetime
from typing import Any, Optional

from shopping_agent.common.types import ShoppingTask, UserProfile
from shopping_agent.storage.profile_store import ProfileStore


class PreferenceMemory:
    def __init__(self, profile_store: Optional[ProfileStore] = None):
        self._store = profile_store or ProfileStore()
        self._implicit_signals: dict[str, list[dict]] = {}

    def load(self, user_id: str,
             task: Optional[ShoppingTask] = None) -> UserProfile:
        """
        加载用户画像。
        若用户无历史记录，返回空白画像（冷启动）。
        """
        profile = self._store.load(user_id)

        # 根据当前任务场景做上下文感知的偏好加权
        if task:
            profile = self._contextualize(profile, task)

        return profile

    def save(self, profile: UserProfile) -> None:
        """持久化用户画像。"""
        self._store.save(profile)

    def update_explicit(
        self,
        user_id: str,
        signal_type: str,       # "purchase", "reject", "add_to_cart"
        product_attrs: dict,    # {"brand": "Sony", "price": 2999, "category": "laptop"}
        delta: float = 0.1,
    ) -> None:
        """
        根据显式行为更新用户画像。
        signal_type="purchase"/"add_to_cart" → 正向更新
        signal_type="reject" → 负向更新
        """
        profile = self.load(user_id)
        direction = 1.0 if signal_type in ("purchase", "add_to_cart") else -1.0
        self._increment_signal_count(profile, signal_type)

        # 更新品牌权重
        brand = product_attrs.get("brand")
        if brand:
            current = profile.brand_weights.get(brand, 0.5)
            profile.brand_weights[brand] = max(0.0, min(1.0,
                                               current + direction * delta))

        # 更新品类亲和度
        category = product_attrs.get("category")
        if category and direction > 0:
            profile.category_affinity[category] = (
                profile.category_affinity.get(category, 0.0) + delta
            )

        # 更新价格敏感度（拒绝高价商品 → 提高敏感度）
        price = product_attrs.get("price")
        if price and signal_type == "reject":
            # 简单启发：拒绝高于均值的商品说明价格敏感
            profile.price_sensitivity = min(1.0,
                                            profile.price_sensitivity + 0.05)
        if price and signal_type == "purchase":
            profile.budget_anchor_history.append(float(price))
            profile.budget_anchor_history = profile.budget_anchor_history[-10:]
            self._record_owned_item(profile, product_attrs)
            self._update_purchase_rhythm(profile, float(price))

        self._update_persona_from_product_signal(profile, product_attrs, direction, delta)
        self._update_long_horizon_profile(profile, product_attrs, direction, signal_type)

        profile.last_updated = datetime.now()
        self._store.log_interaction_signal(
            user_id=user_id,
            signal_type=signal_type,
            product_attrs=product_attrs,
        )
        self.save(profile)

    def update_implicit(
        self,
        user_id: str,
        signal_type: str,       # "dwell", "skip"
        product_attrs: dict,
        dwell_seconds: float = 0.0,
    ) -> None:
        """
        记录隐式行为信号，用于软更新画像。
        """
        signal = {
            "signal_type": signal_type,
            "product_attrs": product_attrs,
            "dwell_seconds": dwell_seconds,
            "timestamp": datetime.now().isoformat(),
        }
        signals = self._implicit_signals.setdefault(user_id, [])
        signals.append(signal)
        self._store.log_interaction_signal(
            user_id=user_id,
            signal_type=signal_type,
            product_attrs=product_attrs,
            dwell_seconds=dwell_seconds,
        )

        # 简单规则：停留超过 30 秒视为正向隐式信号
        if signal_type == "dwell" and dwell_seconds > 30:
            self.update_explicit(user_id, "add_to_cart", product_attrs, delta=0.05)

    def register_persona_drift(
        self,
        user_id: str,
        drift_type: str,
        evidence: str,
        magnitude: float,
        before: Optional[dict[str, Any]] = None,
        after: Optional[dict[str, Any]] = None,
        profile: Optional[UserProfile] = None,
    ) -> None:
        profile = profile or self.load(user_id)
        self._append_drift_event(
            profile,
            drift_type=drift_type,
            evidence=evidence,
            magnitude=magnitude,
            before=before or {},
            after=after or {},
        )
        self.save(profile)

    def get_recent_signals(self, user_id: str, limit: int = 20) -> list[dict]:
        """返回用户近期交互信号，优先读取持久化记录。"""
        persisted = self._store.list_interaction_signals(user_id, limit=limit)
        if persisted:
            return persisted
        return list(reversed(self._implicit_signals.get(user_id, [])))[-limit:]

    def _contextualize(self, profile: UserProfile, task: ShoppingTask) -> UserProfile:
        """
        根据当前任务的隐式需求，临时调整画像权重。
        不修改原始画像，返回调整副本。
        """
        ctx_profile = copy.deepcopy(profile)

        # 出差场景：提高履约时效偏好
        if any("出差" in need or "商务" in need for need in task.implicit_needs):
            ctx_profile.prefer_fast_delivery = True

        # 有硬性预算约束时，临时提高价格敏感度
        budget = task.get_hard_constraints().get("budget_total")
        if budget and budget < 1000:
            ctx_profile.price_sensitivity = min(1.0,
                                                ctx_profile.price_sensitivity + 0.2)

        # 近期正向信号会轻微提升相关品牌/品类偏好
        for signal in self.get_recent_signals(profile.user_id, limit=10):
            signal_type = signal.get("signal_type", "")
            attrs = signal.get("product_attrs", {})
            if signal_type not in {"purchase", "add_to_cart", "dwell"}:
                continue

            brand = attrs.get("brand")
            category = attrs.get("category")
            if brand and (not task.categories or category in task.categories):
                current = ctx_profile.brand_weights.get(brand, 0.5)
                ctx_profile.brand_weights[brand] = min(1.0, current + 0.03)
            if category and (not task.categories or category in task.categories):
                ctx_profile.category_affinity[category] = (
                    ctx_profile.category_affinity.get(category, 0.0) + 0.05
                )

        drift_type = ctx_profile.recent_persona_drift.get("type")
        if drift_type == "budget_drift":
            magnitude = float(ctx_profile.recent_persona_drift.get("magnitude", 0.0))
            ctx_profile.price_sensitivity = max(
                0.0, ctx_profile.price_sensitivity - min(0.12, magnitude * 0.2)
            )
        elif drift_type == "identity_drift":
            target = ctx_profile.recent_persona_drift.get("target")
            if target:
                current = ctx_profile.identity_goal.get(target, 0.0)
                ctx_profile.identity_goal[target] = min(1.0, current + 0.08)
        elif drift_type == "brand_drift":
            current = ctx_profile.brand_orientation.get("brand_signal", 0.0)
            ctx_profile.brand_orientation["brand_signal"] = min(1.0, current + 0.08)
        elif drift_type == "style_drift":
            target = ctx_profile.recent_persona_drift.get("target")
            if target:
                current = ctx_profile.aesthetic_preference.get(target, 0.0)
                ctx_profile.aesthetic_preference[target] = min(1.0, current + 0.08)

        # 基于已有 setup 为当前任务提供“下一步升级”语境
        if task.task_type.value == "bundle" and task.categories:
            for category in task.categories:
                setup_key = f"{category}_setup"
                setup = ctx_profile.active_setups.get(setup_key)
                if not setup:
                    continue
                missing = setup.get("missing", [])
                if missing:
                    ctx_profile.aspiration_signals.append({
                        "source": "setup_gap",
                        "setup_key": setup_key,
                        "missing": missing[:2],
                    })
            ctx_profile.aspiration_signals = ctx_profile.aspiration_signals[-10:]

        return ctx_profile

    @staticmethod
    def _increment_signal_count(profile: UserProfile, signal_type: str) -> None:
        profile.interaction_signal_counts[signal_type] = (
            profile.interaction_signal_counts.get(signal_type, 0) + 1
        )

    def _update_persona_from_product_signal(
        self,
        profile: UserProfile,
        product_attrs: dict,
        direction: float,
        delta: float,
    ) -> None:
        tags = product_attrs.get("persona_tags") or {}
        if not tags:
            return

        style_tags = tags.get("style_signal", [])
        for tag in style_tags[:2]:
            current = profile.aesthetic_preference.get(tag, 0.0)
            profile.aesthetic_preference[tag] = max(
                0.0, min(1.0, current + direction * delta * 0.6)
            )

        for tag in tags.get("identity_fit", [])[:2]:
            current = profile.identity_goal.get(tag, 0.0)
            profile.identity_goal[tag] = max(
                0.0, min(1.0, current + direction * delta * 0.5)
            )

        symbolic_value = tags.get("symbolic_value")
        if symbolic_value == "taste_signaling":
            current = profile.brand_orientation.get("brand_signal", 0.0)
            profile.brand_orientation["brand_signal"] = max(
                0.0, min(1.0, current + direction * delta * 0.5)
            )
        elif symbolic_value == "utilitarian":
            current = profile.brand_orientation.get("function_first", 0.0)
            profile.brand_orientation["function_first"] = max(
                0.0, min(1.0, current + direction * delta * 0.5)
            )

    @staticmethod
    def _append_drift_event(
        profile: UserProfile,
        drift_type: str,
        evidence: str,
        magnitude: float,
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> None:
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": drift_type,
            "evidence": evidence,
            "magnitude": round(max(0.0, magnitude), 3),
            "before": before,
            "after": after,
        }
        if after.get("target"):
            event["target"] = after["target"]
        profile.recent_persona_drift = event
        profile.persona_transition_log.append(event)
        profile.persona_transition_log = profile.persona_transition_log[-20:]
        profile.persona_stability = max(0.1, min(1.0, profile.persona_stability - magnitude * 0.25))

    @staticmethod
    def _record_owned_item(profile: UserProfile, product_attrs: dict[str, Any]) -> None:
        product_id = product_attrs.get("product_id") or product_attrs.get("id")
        if product_id and any(item.get("product_id") == product_id for item in profile.owned_items):
            return

        owned_record = {
            "product_id": product_id or f"owned_{len(profile.owned_items) + 1}",
            "category": product_attrs.get("category", ""),
            "brand": product_attrs.get("brand", ""),
            "price": product_attrs.get("price"),
            "acquired_at": datetime.now().isoformat(),
        }
        profile.owned_items.append(owned_record)
        profile.owned_items = profile.owned_items[-50:]

    @staticmethod
    def _update_purchase_rhythm(profile: UserProfile, price: float) -> None:
        cadence = profile.purchase_rhythm.get("cadence", "steady")
        avg_spend = profile.purchase_rhythm.get("avg_spend", 0.0)
        count = profile.purchase_rhythm.get("purchase_count", 0) + 1
        new_avg = ((avg_spend * (count - 1)) + price) / count
        if count >= 3 and new_avg > 1500:
            cadence = "upgrade_oriented"
        elif count >= 3 and new_avg < 500:
            cadence = "incremental"

        profile.purchase_rhythm.update(
            {
                "avg_spend": round(new_avg, 2),
                "purchase_count": count,
                "cadence": cadence,
            }
        )

    def _update_long_horizon_profile(
        self,
        profile: UserProfile,
        product_attrs: dict[str, Any],
        direction: float,
        signal_type: str,
    ) -> None:
        category = product_attrs.get("category")
        if not category:
            return

        setup_key = f"{category}_setup"
        setup = profile.active_setups.setdefault(
            setup_key,
            {
                "owned": [],
                "missing": [],
                "style": "emerging",
                "next_best_upgrade": "",
            },
        )

        brand = product_attrs.get("brand", "")
        if signal_type in {"purchase", "add_to_cart"} and direction > 0:
            owned_entry = product_attrs.get("product_id") or product_attrs.get("title") or category
            if owned_entry not in setup["owned"]:
                setup["owned"].append(owned_entry)

            complements = self._infer_setup_missing(category)
            setup["missing"] = [item for item in complements if item not in setup["owned"]][:3]
            setup["next_best_upgrade"] = setup["missing"][0] if setup["missing"] else ""
            setup["style"] = self._infer_setup_style(profile, product_attrs)

            owned_count = len(setup["owned"])
            if owned_count <= 1:
                profile.upgrade_stage[setup_key] = "starter"
            elif owned_count == 2:
                profile.upgrade_stage[setup_key] = "growing"
            else:
                profile.upgrade_stage[setup_key] = "refinement"

            if profile.recent_persona_drift:
                profile.aspiration_signals.append(
                    {
                        "source": "recent_drift",
                        "setup_key": setup_key,
                        "drift_type": profile.recent_persona_drift.get("type"),
                        "target": profile.recent_persona_drift.get("target"),
                    }
                )
                profile.aspiration_signals = profile.aspiration_signals[-10:]

    @staticmethod
    def _infer_setup_missing(category: str) -> list[str]:
        mapping = {
            "monitor": ["monitor_arm", "light_bar"],
            "headset": ["stand", "case"],
            "keyboard": ["wrist_rest", "desk_mat"],
            "mouse": ["mouse_pad"],
        }
        return mapping.get(category, [])

    @staticmethod
    def _infer_setup_style(profile: UserProfile, product_attrs: dict[str, Any]) -> str:
        tags = product_attrs.get("persona_tags") or {}
        style_signal = tags.get("style_signal", [])
        if style_signal:
            return style_signal[0]
        if profile.aesthetic_preference:
            return max(profile.aesthetic_preference.items(), key=lambda item: item[1])[0]
        return "practical"
