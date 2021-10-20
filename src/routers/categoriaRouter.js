const CategoriaController = require('../controllers/categoriaController');
//Importar express
const express = require('express');

class CategoriaRouter {

    constructor() {
        this.router = express.Router();
        this.configRouters();
    }

    configRouters() {
        const objCategoriaC = new CategoriaController();
        //Rutas
        this.router.post('/Categoria', objCategoriaC.crearCategoria);
        this.router.get('/Categoria', objCategoriaC.obtenerCategorias);
        this.router.put('/Categoria', objCategoriaC.actualizarCategoria);
        this.router.delete('/Categoria', objCategoriaC.eliminarCategoria);
    }
}

module.exports = categoriaRouter;