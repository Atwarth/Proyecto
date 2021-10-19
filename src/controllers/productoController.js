
const producto = require('../models/producto');

class ProductoController{
    constructor(){

    }

    crearProducto(req, res){
        producto.create(req.body,(error, data)=>{
            if(error){
                res.status(500).json({error});
            }else{
                res.status(201).json(data);
            }
        });
    }

    obtenerProductos(req, res){
        producto.find((error, data)=>{
            if(error){
                res.status(500).json({error});
            }else{
                res.status(201).json(data);
            }
        });

    }

    actualizarProducto(req, res){
        let {id, nombre, precio, categoria, descripcion, urlImagen}= req.body;
        let objProducto = {
            nombre, precio, categoria, descripcion, urlImagen
        }
        producto.findByIdAndUpdate(id, {$set: objProducto}, (error, data) =>{
            if(error){
                res.status(500).json({error});
            }else{
                res.status(200).json(data);
            }

        });
    }

    eliminarProducto(req, res){
        let {id} = req.body;
        producto.findOneAndDelete(id, (error, data) =>{
            if(error){
                res.status(500).json({error});
            }else{
                res.status(200).json(data);
            }
        });
    }
}

module.exports = ProductoController;