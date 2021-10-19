const mongoose = require('mongoose');

const Schema = mongoose.Schema;

let personaSchema = new Schema({
    nombre:{
        type: String
    },
    apellido:{
        type: String
    },
    contrasena:{
        type: String
    },
    correo:{
        type: String
    },
    telefono:{
        type: Number
    },
    fechaNacimiento:{
        type: String
    },
    direccion:{
        type: String
    }
},{
    collection:"persona"
});

module.exports = mongoose.model('Persona',personaSchema);