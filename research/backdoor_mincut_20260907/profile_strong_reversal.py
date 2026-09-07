import cProfile
import io
import json
from pathlib import Path
import pstats
import time

from cpt_world.world_space import WorldGrammar, iter_sampled_seeds

root = Path('/home/chen/runs/strong-reversal-kernel-20260907')
profiler = cProfile.Profile()
start = time.perf_counter()
with profiler:
    tasks = iter_sampled_seeds(WorldGrammar(), query_types=('best_intervention',), count=20)
profiler.dump_stats(root / 'baseline.prof')
stream = io.StringIO()
pstats.Stats(profiler, stream=stream).sort_stats('cumulative').print_stats(35)
(root / 'baseline-profile.txt').write_text(stream.getvalue())
print(json.dumps({'seconds': time.perf_counter()-start, 'count':len(tasks),'ids':[s['seed_id'] for s in tasks]}))
print(stream.getvalue())
