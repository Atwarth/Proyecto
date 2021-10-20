const categoria = require('../models/categoria');


class CategoriaController {
    constructor() {

    }
    // Req -- informacion del cliente -- Res es la respuesta del servidor
    crearCategoria(req, res) {
        categoria.create(req.body, (error, data) => {
            if (error) {
                res.status(500).json({ error });
            } else {
                res.status(201).json(data);
            }
        });
    }

    obtenerProductos(req, res) {
        categoria.find((error, data) => {
            if (error) {
                res.status(500).json({ error });
            } else {
                res.status(200).json(data);
            }
        });
    }

    actualizarCategoria(req, res) {
        let { empresa,nombreDeLaMision,naveEspacial,tiempoDeVuelo,altitudMaximaDelVuelo,velocidadMaxima,asientosDisponibles,precioDelVuelo,fechaDeDespegue} = req.body;
        let objProducto = {
            empresa,nombreDeLaMision,naveEspacial,tiempoDeVuelo,altitudMaximaDelVuelo,velocidadMaxima,asientosDisponibles,precioDelVuelo,fechaDeDespegue
        }
        categoria(id, { $set: objProducto }, (error, data) => {
            if (error) {
                res.status(500).json({ error });
            } else {
                res.status(200).json(data);
            }
        });
    }

    eliminarCategoria(req, res) {
        let { id } = req.body;
        categoria.findOneAndRemove(id, (error, data) => {
            if (error) {
                res.status(500).json({ error });
            } else {
                res.status(200).json(data);
            }
        });
    }
}

module.exports = CategoriaController;