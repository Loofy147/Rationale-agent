import re
from mas.data_models import AdaptivePlan, AdaptiveFeature, AdaptiveTask

class PlanParserService:
    """
    Parses a markdown string into a structured AdaptivePlan object.
    """
    def parse(self, markdown_text: str) -> AdaptivePlan | None:
        try:
            epic_title = re.search(r"# Epic: (.+)", markdown_text).group(1).strip()

            features_text = markdown_text.split("## Feature: ")[1:]
            features = []
            for i, feature_text in enumerate(features_text):
                feature_title = feature_text.split("\n")[0].strip()

                # Find the table for this feature
                table_text = feature_text.split("| ID |")[1]

                tasks = []
                # Regex to capture the content of each row in the markdown table
                task_rows = re.findall(r"\| (T-\d+) \| (.+?) \| (.+?) \| (.+?) \| (.+?) \| (.+?) \| (.+?) \| (.+?) \|", table_text)

                for row in task_rows:
                    tasks.append(AdaptiveTask(
                        id=row[0].strip(),
                        type=row[1].strip(),
                        title=row[2].strip(),
                        description=row[3].strip(),
                        estimate=row[4].strip(),
                        priority=row[5].strip(),
                        risk=row[6].strip(),
                    ))

                features.append(AdaptiveFeature(
                    title=feature_title,
                    tasks=tasks
                ))

            return AdaptivePlan(epic_title=epic_title, features=features)

        except (AttributeError, IndexError) as e:
            print(f"Error parsing plan: {e}")
            # In a real app, you would log this error.
            return None

plan_parser_service = PlanParserService()
