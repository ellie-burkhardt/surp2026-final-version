import numpy as np


class ForecastGenerator:

    def __init__(self):

        # These are placeholders until we calibrate
        # against historical forecast distributions.

        self.active_threshold = 0.01

        self.problem_sizes = {
            "storm_slab": "D2",
            "wind_slab": "D2",
            "wet_loose": "D1",
            "persistent_slab": "D1",
        }

        self.problem_display_names = {
            "storm_slab": "Storm Slab",
            "wind_slab": "Wind Slab",
            "wet_loose": "Wet Loose",
            "persistent_slab": "Persistent Slab",
        }

    def generate(
        self,
        predictions,
    ):

        problems = []

        for problem_name, probs in predictions.items():

            problem = self._build_problem(
                problem_name,
                probs,
            )

            problems.append(problem)

        overall_danger = (
            self._compute_overall_danger(
                problems
            )
        )

        discussion = (
            self._build_discussion(
                overall_danger,
                problems,
            )
        )

        return {
            "overall_danger": overall_danger,
            "problems": problems,
            "discussion": discussion,
        }

    def _build_problem(
        self,
        problem_name,
        probabilities,
    ):

        values = np.array(
            list(probabilities.values())
        )

        max_probability = float(
            values.max()
        )

        mean_probability = float(
            values.mean()
        )

        dominant_aspects = (
            self._identify_aspects(
                probabilities
            )
        )

        active = (
            max_probability
            >= self.active_threshold
        )

        if not active:
            dominant_aspects = []

        likelihood = (
            self._assign_likelihood(
                max_probability
            )
        )

        danger = (
            self._assign_danger(
                max_probability
            )
        )

        return {
            "name":
                self.problem_display_names[
                    problem_name
                ],

            "problem_type":
                problem_name,

            "active":
                active,

            "max_probability":
                max_probability,

            "mean_probability":
                mean_probability,

            "dominant_aspects":
                dominant_aspects,

            "likelihood":
                likelihood,

            "danger":
                danger,

            "size":
                self.problem_sizes[
                    problem_name
                ],
        }

    def _identify_aspects(
        self,
        probabilities,
    ):

        max_prob = max(
            probabilities.values()
        )

        threshold = (
            max_prob * 0.75
        )

        aspects = []

        for target, prob in probabilities.items():

            if prob < threshold:
                continue

            aspect = (
                target.split("_")[-1]
            )

            aspects.append(
                aspect
            )

        return aspects

    def _assign_likelihood(
        self,
        probability,
    ):

        if probability < 0.01:
            return "Unlikely"

        elif probability < 0.05:
            return "Possible"

        elif probability < 0.20:
            return "Likely"

        return "Very Likely"

    def _assign_danger(
        self,
        probability,
    ):

        if probability < 0.01:
            return {
                "rating": "Low",
                "score": 1,
            }

        elif probability < 0.05:
            return {
                "rating": "Moderate",
                "score": 2,
            }

        elif probability < 0.20:
            return {
                "rating": "Considerable",
                "score": 3,
            }

        elif probability < 0.50:
            return {
                "rating": "High",
                "score": 4,
            }

        return {
            "rating": "Extreme",
            "score": 5,
        }

    def _compute_overall_danger(
        self,
        problems,
    ):

        max_score = max(
            p["danger"]["score"]
            for p in problems
        )

        lookup = {
            1: "Low",
            2: "Moderate",
            3: "Considerable",
            4: "High",
            5: "Extreme",
        }

        return {
            "rating":
                lookup[max_score],
            "score":
                max_score,
        }

    def _build_discussion(
        self,
        overall_danger,
        problems,
    ):

        lines = []

        lines.append(
            "AVALANCHE FORECAST"
        )

        lines.append("")

        lines.append(
            f"Overall Danger: "
            f"{overall_danger['rating']}"
        )

        lines.append("")

        for problem in problems:

            if not problem["active"]:
                continue

            aspects = (
                ", ".join(
                    problem[
                        "dominant_aspects"
                    ]
                )
            )

            lines.append(
                f"{problem["name"]}:"
            )

            lines.append(
                f"{problem['name']} avalanches "
                f"are {problem['likelihood'].lower()} "
                f"on {aspects} aspects."
            )

            lines.append(
                f"Expected size: "
                f"{problem['size']}."
            )

            lines.append(
                f"Danger Rating: "
                f"{problem['danger']['rating']}."
            )

            lines.append("")

        if all(
            not p["active"]
            for p in problems
        ):

            lines.append(
                "No significant avalanche "
                "problems identified."
            )

        return "\n".join(
            lines
        )

    def to_text(
        self,
        forecast,
    ):

        return forecast[
            "discussion"
        ]