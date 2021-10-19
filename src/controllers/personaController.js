const persona = require('../models/persona');

class PersonaController{
    constructor(){

    }

    crearPersona(req,res){
        persona.create(req.body,(error,data)=>{
            if(error){
                res.status(500).json({error});
            }else{
                res.status(201).json(data);
            }
        });
    }

    obtenerPersona(req,res){
        persona.find((error,data)=>{
            if(error){
                res.status(500).json({error});
            }else{
                res.status(200).json(data);
            }
        });
    }

    actualizarPersona(req,res){
        let{ id, nombre, apellido, contrasena, telefono, direccion, fechaNacimiento} =req.body;
        let objPersona ={
            nombre,apellido, contrasena, telefono, direccion, fechaNacimiento
        };
        persona.findByIdAndUpdate(id,{$set: objPersona},(error,data) =>{
            if(error){
                res.status(500).json({error});
            }else{
                res.status(200).json(data);
            }
        });
    }

    eliminarPersona(req,res){
        let {id} = req.body;
        persona.findByIdAndRemove(id,(error,data)=>{
            if(error){
                res.status(500).json({error});
            }else{
                res.status(200).json(data);
            }
        });
    }

}

module.exports = PersonaController;