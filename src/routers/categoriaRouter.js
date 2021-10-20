const express = require('express');
const CategoriaController = require('../controllers/categoriaController');

class CategoriaRouter {

    constructor() {
        this.router = express.Router();
        this.configRouters();
    }

    configRouters() {
        const objCategoriaC = new CategoriaController();
        //Rutas
        this.router.post('/categoria', objCategoriaC.crearCategoria);
        this.router.get('/categoria', objCategoriaC.obtenerCategorias);
        this.router.put('/categoria', objCategoriaC.actualizarCategoria);
        this.router.delete('/categoria', objCategoriaC.eliminarCategoria);
    }
}

module.exports = CategoriaRouter;