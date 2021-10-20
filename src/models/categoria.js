const mongoose = require('mongoose');

const Schema = mongoose.Schema;

let categoriaSchema = new Schema({
    empresa :{
        type: String
    },
    nombreDeLaMision :{ 
        type: String
    },
    naveEspacial :{ 
        type: String
    },
    tiempoDeVuelo :{ 
        // En minutos
        type: Number
    },
    altitudMaximaDelVuelo :{ 
        // En kilometros por hora
        type: Number
    },
    velocidadMaxima :{ 
        // Veklocidad en Kilometros por hora
        type: Number
    },
    asientosDisponibles :{ 
        type: Number
    },
    precioDelVuelo :{ 
        type: Number
    },
    fechaDeDespegue :{
        type: Date
    },
}, {
    collection: "categorias"
});

module.exports = mongoose.model("Categoria", categoriaSchema);