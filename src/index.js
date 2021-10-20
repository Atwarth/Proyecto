//dependencias
const express = require('express');
const mongoose = require('mongoose')
const cors = require('cors');

//importar modulos para cada modelo
const key_database = require('./database/key_database');
const PersonaRouter= require('./routers/personaRouter');
const ProductoRouter = require('./routers/productoRouter');
const CategoriaRouter = require('./routers/categoriaRouter');

class Server{
    constructor(){
        this.conectarBd();
        this.app = express();
        this.config();
    }

    config(){
        this.app.set('port', process.env.PORT || 3000);
        this.app.use(express.json());
        //Añadimos cors para permitir conexiones de origen cruzado
        this.app.use(cors());
        //Crear ruta raiz del servidor 
        let router = express.Router();
        router.get('/',(req,res)=>{
            res.status(200).json({"message":"all ok"});
        });
        //añadir ruta a express
        this.app.use(router);
        // crear rutas diferentes a la raiz una para cada modelo
        let objPersonaR = new PersonaRouter();
        this.app.use(objPersonaR.router);
        let objProductoR = new ProductoRouter();
        this.app.use(objProductoR.router);
        let objCategoriaR = new CategoriaRouter();
        this.app.use(objCategoriaR.router);
        


        
        //poner el servidor a la escucha
        this.app.listen(this.app.get('port'),()=>{
            console.log("Servidor corriendo por el puerto =>", this.app.get('port'));
        });
    }

    conectarBd(){
        mongoose.connect(key_database.db).then(()=>{
            console.log("conexion exitosa a la BD");
        }).catch(error=>{
            console.error(error);
        });
    }
}

new Server();