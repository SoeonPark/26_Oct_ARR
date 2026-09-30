import torch
from torch.utils.data import Sampler


def shuffled_batches(start, stop, batch_size, seed):
    """Repeatedly shuffle one index range and yield full batches only."""
    size = stop - start
    if size < batch_size:
        raise ValueError(
            f"Pool size {size} is smaller than batch size {batch_size}."
        )

    generator = torch.Generator().manual_seed(seed)
    while True:
        order = torch.randperm(size, generator=generator).tolist()
        for pos in range(0, size - batch_size + 1, batch_size):
            yield [start + i for i in order[pos:pos + batch_size]]


class PairBatchSampler(Sampler):
    """Yield batches of (alignment_index, downstream_index) tuples.

    One iteration describes the full, single-process max_steps training run.
    Every iteration recreates the same plan so Trainer can skip consumed
    batches when resuming. Objective selection belongs to the Trainer callback,
    which receives the planned optimizer step, never the live global_step.
    """

    def __init__(
        self,
        pair_ranges,
        downstream_size,
        batch_size,
        num_steps,
        accumulation_steps,
        seed,
        objective_at,
    ):
        if batch_size < 2 or num_steps <= 0 or accumulation_steps <= 0:
            raise ValueError(
                "Need batch_size >= 2 and positive step counts."
            )
        if not pair_ranges:
            raise ValueError("No alignment language pairs were loaded.")

        self.pair_ranges = dict(pair_ranges)
        self.downstream_size = downstream_size
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.accumulation_steps = accumulation_steps
        self.seed = seed
        self.objective_at = objective_at
        self.drop_last = True

    def __len__(self):
        return self.num_steps * self.accumulation_steps

    def __iter__(self):
        pairs = sorted(self.pair_ranges)
        pair_generator = torch.Generator().manual_seed(self.seed)
        # Independent RNGs keep each objective's sample sequence unchanged
        # when another objective is interleaved or moved to a separate phase.
        alignment_streams = {
            pair: shuffled_batches(
                *self.pair_ranges[pair],
                self.batch_size,
                self.seed + 2 + i,
            )
            for i, pair in enumerate(pairs)
        }
        downstream_stream = shuffled_batches(
            0,
            self.downstream_size,
            self.batch_size,
            self.seed + 1,
        )
        alignment_updates = 0
        pair_order = []

        for step in range(self.num_steps):
            objective = self.objective_at(step)

            if objective == "alignment":
                slot = alignment_updates % len(pairs)
                if slot == 0:
                    pair_order = torch.randperm(
                        len(pairs), generator=pair_generator,
                    ).tolist()
                pair = pairs[pair_order[slot]]

                # Keep the pair fixed throughout one optimizer update.
                # InfoNCE negatives still come from each microbatch alone.
                for _ in range(self.accumulation_steps):
                    yield [(i, None) for i in next(alignment_streams[pair])]
                alignment_updates += 1

            elif objective == "downstream":
                for _ in range(self.accumulation_steps):
                    yield [(None, i) for i in next(downstream_stream)]

            else:
                raise ValueError(f"Unknown objective: {objective}")


class AlignmentEvalBatchSampler(Sampler):
    """Visit every example once in deterministic, single-pair batches.

    A singleton tail joins the preceding batch, so an actual batch may have
    batch_size + 1 examples. No example is dropped or duplicated.
    """

    def __init__(self, pair_ranges, batch_size):
        if batch_size < 2:
            raise ValueError("Gap evaluation requires batch_size >= 2.")
        if not pair_ranges:
            raise ValueError("No alignment language pairs were loaded.")
        self.pair_ranges = dict(pair_ranges)
        self.batch_size = batch_size
        self.drop_last = False
        for pair, (start, stop) in self.pair_ranges.items():
            if start < 0 or stop - start < 2:
                raise ValueError(
                    f"Gap evaluation needs at least two examples for {pair}: "
                    f"range=({start}, {stop})."
                )

    def __len__(self):
        count = 0
        for start, stop in self.pair_ranges.values():
            size = stop - start
            count += (size + self.batch_size - 1) // self.batch_size
            if size > self.batch_size and size % self.batch_size == 1:
                count -= 1
        return count

    def __iter__(self):
        for pair in sorted(self.pair_ranges):
            start, stop = self.pair_ranges[pair]
            while start < stop:
                end = min(start + self.batch_size, stop)
                if stop - end == 1:
                    end = stop
                yield list(range(start, end))
                start = end
