const tf = require("@tensorflow/tfjs-node");
const InputError = require('../exceptions/InputError');

async function predictClassification(model, image) {
    try {
        const tensor = tf.node
          .decodeJpeg(image)
          .resizeNearestNeighbor([224, 224])
          .expandDims()
          .toFloat();
      
        const prediction = model.predict(tensor);
        const score = await prediction.data();
        const confidenceScore = score[0]; 

        const label = confidenceScore > 0.5 ? 'Cancer' : 'Non-cancer';
      
        let suggestion;
      
        if (label === 'Cancer') {
          suggestion = 'Segera periksa ke dokter!'
        }
      
        if (label === 'Non-cancer') {
          suggestion = 'Tidak perlu khawatir, ini bukan cancer!'
        }
      
        return { label, suggestion };
    } catch (e) {
        throw new InputError(`Terjadi kesalahan dalam melakukan prediksi`)
    }
}

module.exports = predictClassification;
