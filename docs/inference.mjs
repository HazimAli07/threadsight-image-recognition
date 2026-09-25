// Inference from the exact trained ThreadSight weights, entirely in the browser.
export function createModel(manifest, buffer) {
  const floats = new Float32Array(buffer);
  const arrays = {};
  for (const [name, spec] of Object.entries(manifest.arrays)) {
    const size = spec.shape.reduce((product, dimension) => product * dimension, 1);
    arrays[name] = floats.subarray(spec.offset, spec.offset + size);
    if (arrays[name].length !== size) throw new Error(`Incomplete model weights: ${name}`);
  }
  return {classes: manifest.classes, arrays};
}

function softmax(logits) {
  const max = Math.max(...logits);
  const scores = logits.map((value) => Math.exp(value - max));
  const total = scores.reduce((sum, value) => sum + value, 0);
  return scores.map((value) => value / total);
}

function classicalScores(input, weights) {
  const {ml_mean: mean, ml_scale: scale, ml_coef: coef, ml_intercept: intercept} = weights;
  const probabilities = [];
  for (let category = 0; category < 10; category++) {
    let logit = intercept[category];
    const row = category * 784;
    for (let pixel = 0; pixel < 784; pixel++) {
      logit += coef[row + pixel] * ((input[pixel] - mean[pixel]) / scale[pixel]);
    }
    // SGDClassifier's multiclass log_loss uses one-vs-rest sigmoid scores,
    // normalized across classes, rather than a multiclass softmax.
    probabilities.push(Math.fround(1 / (1 + Math.exp(-logit))));
  }
  const total = probabilities.reduce((sum, value) => sum + value, 0);
  if (total === 0) return probabilities.map(() => 1 / probabilities.length);
  return probabilities.map((value) => value / total);
}

function convolution(input, inputChannels, side, weights, bias, outputChannels) {
  const output = new Float32Array(outputChannels * side * side);
  const plane = side * side;
  for (let outChannel = 0; outChannel < outputChannels; outChannel++) {
    for (let row = 0; row < side; row++) {
      for (let column = 0; column < side; column++) {
        let value = bias[outChannel];
        for (let inChannel = 0; inChannel < inputChannels; inChannel++) {
          const kernel = (outChannel * inputChannels + inChannel) * 9;
          const source = inChannel * plane;
          for (let dr = -1; dr <= 1; dr++) {
            const y = row + dr;
            if (y < 0 || y >= side) continue;
            for (let dc = -1; dc <= 1; dc++) {
              const x = column + dc;
              if (x < 0 || x >= side) continue;
              value += input[source + y * side + x] * weights[kernel + (dr + 1) * 3 + dc + 1];
            }
          }
        }
        output[outChannel * plane + row * side + column] = Math.max(0, value);
      }
    }
  }
  return output;
}

function maxPool(input, channels, side) {
  const outSide = side / 2;
  const output = new Float32Array(channels * outSide * outSide);
  for (let channel = 0; channel < channels; channel++) {
    const source = channel * side * side;
    const target = channel * outSide * outSide;
    for (let row = 0; row < outSide; row++) {
      for (let column = 0; column < outSide; column++) {
        const index = source + row * 2 * side + column * 2;
        output[target + row * outSide + column] = Math.max(
          input[index], input[index + 1], input[index + side], input[index + side + 1]
        );
      }
    }
  }
  return output;
}

function dense(input, weights, bias, outputs, relu) {
  const output = new Float32Array(outputs);
  for (let neuron = 0; neuron < outputs; neuron++) {
    let value = bias[neuron];
    const row = neuron * input.length;
    for (let index = 0; index < input.length; index++) value += input[index] * weights[row + index];
    output[neuron] = relu ? Math.max(0, value) : value;
  }
  return output;
}

function cnnScores(input, weights) {
  const first = convolution(input, 1, 28, weights.conv1_weight, weights.conv1_bias, 32);
  const firstPool = maxPool(first, 32, 28);
  const second = convolution(firstPool, 32, 14, weights.conv2_weight, weights.conv2_bias, 64);
  const secondPool = maxPool(second, 64, 14);
  const hidden = dense(secondPool, weights.dense1_weight, weights.dense1_bias, 128, true);
  return softmax([...dense(hidden, weights.dense2_weight, weights.dense2_bias, 10, false)]);
}

function summarize(scores, classes) {
  const indices = [...scores.keys()].sort((left, right) => {
    const difference = scores[right] - scores[left];
    return Math.abs(difference) < 1e-7 ? right - left : difference;
  }).slice(0, 3);
  return {
    label: classes[indices[0]],
    score: scores[indices[0]],
    top_three: indices.map((index) => ({label: classes[index], score: scores[index]})),
  };
}

export function predictPixels(pixels, model) {
  if (pixels.length !== 784) throw new Error("Expected a 28 × 28 image.");
  const input = Float32Array.from(pixels, (value) => value / 255);
  return {
    ml: summarize(classicalScores(input, model.arrays), model.classes),
    dl: summarize(cnnScores(input, model.arrays), model.classes),
  };
}
