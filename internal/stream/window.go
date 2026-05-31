package stream

import "github.com/somoprovo/trainpulse/internal/model"

type Window struct {
	frames []model.TelemetryFrame
	size   int
}

func NewWindow(size int) *Window {
	if size < 2 {
		size = 2
	}
	return &Window{size: size, frames: make([]model.TelemetryFrame, 0, size)}
}

func (w *Window) Add(frame model.TelemetryFrame) {
	if len(w.frames) == w.size {
		copy(w.frames, w.frames[1:])
		w.frames[len(w.frames)-1] = frame
		return
	}
	w.frames = append(w.frames, frame)
}

func (w *Window) Frames() []model.TelemetryFrame {
	out := make([]model.TelemetryFrame, len(w.frames))
	copy(out, w.frames)
	return out
}

func AvgGPUUtil(frames []model.TelemetryFrame) float64 {
	var sum float64
	var count int
	for _, f := range frames {
		for _, g := range f.GPUs {
			sum += g.Utilization
			count++
		}
	}
	if count == 0 {
		return 0
	}
	return sum / float64(count)
}

func AvgStepMS(frames []model.TelemetryFrame) float64 {
	var sum float64
	var count int
	for _, f := range frames {
		if f.Training != nil && f.Training.StepTimeMS > 0 {
			sum += f.Training.StepTimeMS
			count++
		}
	}
	if count == 0 {
		return 0
	}
	return sum / float64(count)
}

func AvgTokensPerSec(frames []model.TelemetryFrame) float64 {
	var sum float64
	var count int
	for _, f := range frames {
		if f.Training != nil && f.Training.TokensPerSec > 0 {
			sum += f.Training.TokensPerSec
			count++
		}
	}
	if count == 0 {
		return 0
	}
	return sum / float64(count)
}
