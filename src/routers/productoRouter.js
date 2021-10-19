const express = require("express");
const ProductoController = require("../controllers/productoController");
   

class ProductoRouter{

    constructor(){
        this.router = express.Router();
    }

    config(){
        const objProductoC = new ProductoController();
        this.router.post("/producto", objProductoC.crearProducto);
        this.router.get("/producto", objProductoC.obtenerProductos);
        this.router.put("/producto", objProductoC.actualizarProducto);
        this.router.delete("/producto", objProductoC.eliminarProducto);
    }
}

module.exports = ProductoRouter;