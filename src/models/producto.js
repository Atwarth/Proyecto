const mongoose = require("mongoose");

const Schema = mongoose.Schema;

let productoSchema = new Schema({
    nombre: {
        type: String
    },
    precio: {
        type: Number
    },
    categoria: {
        type: String
    },
    descripcion: {
        type: String
    },
    urlImagen: {
        type: String
    }
},  {
    collection: "productos"
});

module.exports = mongoose.model("Producto", productoSchema);