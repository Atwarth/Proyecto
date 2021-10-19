const express = require('express');
const PersonaController = require('../controllers/personaController');

class PersonaRouter{
    constructor(){
        this.router = express.Router();
        this.config();
    }

    config(){
        const objPersonaC = new PersonaController();
        this.router.post('/persona',objPersonaC.crearPersona);
        this.router.get('/persona', objPersonaC.obtenerPersona);
        this.router.put('/persona',objPersonaC.actualizarPersona);
        this.router.delete('/persona', objPersonaC.eliminarPersona);
    }

}
module.exports = PersonaRouter;