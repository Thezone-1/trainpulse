package collector

import (
	"context"
	"math"
	"time"

	"github.com/somoprovo/trainpulse/internal/model"
)

type SimCollector struct {
	start time.Time
}

func NewSimCollector() *SimCollector {
	return &SimCollector{start: time.Now()}
}

func (s *SimCollector) Name() string { return "sim" }

func (s *SimCollector) Collect(context.Context) (model.TelemetryFrame, error) {
	now := time.Now()
	t := now.Sub(s.start).Seconds()
	phase := math.Mod(t, 80)

	util := 91.0 + 4*math.Sin(t/3)
	temp := 66.0 + 3*math.Sin(t/6)
	mem := uint64(17500 + 300*math.Sin(t/8))
	step := 145.0 + 8*math.Sin(t/5)
	throughput := 220.0 + 10*math.Sin(t/4)
	tokensPerSec := 78000.0 + 2500*math.Sin(t/4)
	mfu := 0.47 + 0.04*math.Sin(t/5)
	memBandwidth := 0.72 + 0.04*math.Sin(t/6)
	tokenizerWait := 4.0
	allReduceWait := 8.0
	pipelineBubble := 7.0
	checkpoint := 0.0
	avgSeqLen := 1850.0
	maxSeqLen := 2048
	dataWait := 8.0
	syncWait := 5.0

	switch {
	case phase > 18 && phase <= 34:
		util = 39 + 8*math.Sin(t)
		step = 260 + 35*math.Sin(t/2)
		throughput = 110 + 8*math.Sin(t)
		tokensPerSec = 36000 + 2500*math.Sin(t)
		mfu = 0.22
		tokenizerWait = 72 + 16*math.Sin(t/3)
		dataWait = 95 + 15*math.Sin(t/3)
	case phase > 34 && phase <= 50:
		util = 72 + 22*math.Sin(t*1.7)
		syncWait = 85 + 25*math.Sin(t/2)
		allReduceWait = 78 + 18*math.Sin(t/2)
		step = 230 + 45*math.Sin(t/2)
		tokensPerSec = 52000 + 3500*math.Sin(t)
	case phase > 50 && phase <= 64:
		mem = uint64(21000 + 1100*math.Sin(t/4) + (phase-50)*260)
		step = 200 + 20*math.Sin(t)
		memBandwidth = 0.97
		avgSeqLen = 730
		checkpoint = 80
	case phase > 64:
		temp = 86 + 4*math.Sin(t)
		util = 62 + 7*math.Sin(t)
		step = 245 + 20*math.Sin(t/2)
		tokensPerSec = 47000 + 2500*math.Sin(t/2)
		pipelineBubble = 62
	}

	gpus := []model.GPUSample{
		simGPU(0, "Simulated H100", util, mem, temp, now),
		simGPU(1, "Simulated H100", util-3+6*math.Sin(t/7), mem+180, temp-1, now),
	}
	if phase > 34 && phase <= 50 {
		gpus[1].Utilization = 34 + 8*math.Sin(t)
	}

	return model.TelemetryFrame{
		Timestamp: now,
		Host: model.HostSample{
			CPUUtilization: 58 + 12*math.Sin(t/5),
			MemoryUsedMB:   36000,
			MemoryTotalMB:  64000,
			Load1:          6 + math.Sin(t/8),
			Timestamp:      now,
		},
		GPUs: gpus,
		Training: &model.TrainingSample{
			WorkloadKind:       "llm_pretraining",
			ModelFamily:        "llama",
			ModelName:          "sim-llama-7b",
			Framework:          "pytorch",
			Precision:          "bf16",
			StepTimeMS:         step,
			Throughput:         throughput,
			TokensPerSec:       tokensPerSec,
			MFU:                mfu,
			TFLOPs:             620 * mfu,
			MemBandwidth:       memBandwidth,
			AvgSeqLen:          avgSeqLen,
			MaxSeqLen:          maxSeqLen,
			DataWaitMS:         dataWait,
			TokenizerWaitMS:    tokenizerWait,
			SyncWaitMS:         syncWait,
			AllReduceWaitMS:    allReduceWait,
			PipelineBubbleMS:   pipelineBubble,
			CheckpointMS:       checkpoint,
			BatchSize:          128,
			MicroBatchSize:     4,
			GradAccumSteps:     16,
			WorldSize:          2,
			TensorParallelSize: 1,
			PipelineStages:     1,
			DataParallelSize:   2,
			Ranks: []model.RankSample{
				{Rank: 0, GPUIndex: 0, StepTimeMS: step, TokensPerSec: tokensPerSec / 2, DataWaitMS: dataWait, SyncWaitMS: syncWait},
				{Rank: 1, GPUIndex: 1, StepTimeMS: step * rankSlowdown(phase, t), TokensPerSec: tokensPerSec / 2.2, DataWaitMS: dataWait / 2, SyncWaitMS: syncWait},
			},
			GlobalStep: int64(t * 6),
			Timestamp:  now,
		},
	}, nil
}

func rankSlowdown(phase, t float64) float64 {
	if phase > 34 && phase <= 50 {
		return 1.48 + 0.08*math.Sin(t)
	}
	return 1.02
}

func simGPU(index int, name string, util float64, mem uint64, temp float64, ts time.Time) model.GPUSample {
	return model.GPUSample{
		Index:       index,
		Name:        name,
		UUID:        "SIM-GPU",
		Utilization: clamp(util, 0, 100),
		MemoryUsed:  mem,
		MemoryTotal: 24576,
		Temperature: clamp(temp, 25, 95),
		PowerWatts:  450 * clamp(util, 0, 100) / 100,
		SMClockMHz:  1410 - max(0, temp-82)*18,
		Timestamp:   ts,
	}
}

func clamp(v, low, high float64) float64 {
	if v < low {
		return low
	}
	if v > high {
		return high
	}
	return v
}

func max(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
