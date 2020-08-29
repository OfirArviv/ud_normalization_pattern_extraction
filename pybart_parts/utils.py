from collections import defaultdict
from typing import List, Dict, Tuple

from pybart_parts.definitions import Extraction
from pybart_parts.gen_pattern import PatternGenerator
from pybart_parts.sample_types import AnnotatedSample

AdvancedPatternStr = str
SimplePatternStr = str


class GenerationFromAnnotatedSamples:
    @staticmethod
    def gen_pattern_dict(
            ann_samples: List[AnnotatedSample], trigger_pattern_generator: PatternGenerator,
            no_trigger_pattern_generator: PatternGenerator
    ) -> Tuple[Dict[AdvancedPatternStr, List[Tuple[Extraction, SimplePatternStr]]],List[str]]:
        error_list = []
        pattern_to_samples = defaultdict(list)
        for sample in ann_samples:
            try:
                if sample.trigger_tok_offsets:
                    pattern = trigger_pattern_generator.generate_trigger_pattern(sample)
                else:
                    pattern = no_trigger_pattern_generator.generate_subject_object_pattern(sample)
                pattern = "| " + "\n  ".join(pattern.splitlines())

                pattern_to_samples[pattern].append(sample)

            except Exception as e:
                # print("Couldn't create pattern: " + str(e))
                error_list.append(str(e))

        return pattern_to_samples, error_list

    @staticmethod
    def prepare_patterns_for_display(patterns):
        patterns = sorted(patterns.items(), key=lambda kv: len(kv[1]), reverse=True)

        out_patterns: List[str] = []
        out_simple_patterns: List[str] = []
        out_counts: List[int] = []
        for pattern, samples in patterns:
            representative_example = min(samples, key=lambda s: len(s.tokens))
            out_patterns.append(pattern)
            out_simple_patterns.append(representative_example.raw)
            out_counts.append(len(samples))

        return out_patterns, out_simple_patterns, out_counts
