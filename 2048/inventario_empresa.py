#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class Inventario:
    def __init__(self):
        
        # Clase que contiene dos inventarios: Materia prima y venta
        
        self.inventario_materia_prima = {
            'envases': 0,
            'etiquetas': 0,
            'melatonina_pura': 0,
            'estearato_magnesio': 0,
            'celulosa_microcristalina': 0
        }
        self.inventario_venta = {
            'envase_melatonina': 0,
            'pastilla_melatonina': 0
        }
        
    # Metodo para realizar el pedido de materia prima
    
    def realizar_pedido_materia_prima(self, producto, cantidad, costo):
        self.inventario_materia_prima[producto] += cantidad
        print(f"Pedido recibido. {cantidad:,.2f} {producto} agregados al inventario. Costo total: ${costo*cantidad:,.2f}\n")

        
    # Metodo para revisar los elementos en los inventarios
    
    def revisar_inventario(self):
        print("Inventario de materia prima:")
        for producto, cantidad in self.inventario_materia_prima.items():
            print(f"{producto}: {cantidad:,.2f}")

        print("\nInventario de venta:")
        for producto, cantidad in self.inventario_venta.items():
            print(f"{producto}: {cantidad:,.2f}")
        print("\n")
    
    # Metodo que convierte los elementos en el inventario de materia prima a elementos de venta
    
    def convertir_a_venta(self, productos_a_convertir):
        # Condicional para comprar que se cumpla los minimos requisitos presentados por el ejercicio
        i = 0
        while i<productos_a_convertir:
            if (
                self.inventario_materia_prima['envases'] >= 1
                and self.inventario_materia_prima['etiquetas'] >= 1
                and self.inventario_materia_prima['melatonina_pura'] >= 0.01 * 90
                and self.inventario_materia_prima['estearato_magnesio'] >= 0.0128 * 90
                and self.inventario_materia_prima['celulosa_microcristalina'] >= 0.2052 * 90
            ):
                self.inventario_materia_prima['envases'] -= 1
                self.inventario_materia_prima['etiquetas'] -= 1
                self.inventario_materia_prima['melatonina_pura'] -= 0.01 * 90
                self.inventario_materia_prima['estearato_magnesio'] -= 0.0128 * 90
                self.inventario_materia_prima['celulosa_microcristalina'] -= 0.2052 * 90
                self.inventario_venta['envase_melatonina'] += 1
                self.inventario_venta['pastilla_melatonina'] += 90
                
                i+=1
            else:
                break
        if i == 0:
            print(f"No hay suficientes materiales. Se ha convertido un total de {i} productos")
        else:
            print(f"Se ha convertido un total de: {i} productos")

    # Metodo que calcula los pedidos proyectados para una cantidad de ventas
    
    def calcular_pedidos_proyectados(self, ventas_proyectadas):
        
        pedidos_necesarios_sin_inventario = {
            'envases': ventas_proyectadas,
            'etiquetas': ventas_proyectadas,
            'melatonina_pura': ventas_proyectadas * 90 * 0.01,
            'estearato_magnesio': ventas_proyectadas * 90 * 0.0128,
            'celulosa_microcristalina': ventas_proyectadas * 90 * 0.2052
        }
        
        pedidos_necesarios_inventario = {elemento: abs(pedidos_necesarios_sin_inventario[elemento] -  self.inventario_materia_prima[elemento]) for elemento in  self.inventario_materia_prima}
        
        return pedidos_necesarios_inventario


# Ejemplo de uso:

empresa = Inventario()

while True:
    print("Menu - Inventario\n")
    print("1. Realizar un pedido")
    print("2. Consultar los inventarios")
    print("3. Convertir los productos del inventario materia prima a los productos de venta")
    print("4. Calcular cuantos pedidos se deben hacer de cada producto segun las ventas proyectadas para el siguiente mes")
    print("5. Salir\n")
    
    seleccion = int(input("Ingrese su eleccion: "))
    
    if seleccion == 1: 
        
        #Realizar pedido de materia prima
        
        envases = int(input("Ingrese el numero de envases. Tenga en cuenta que el numero debe ser mayor a o igual 500: "))
        while envases<500:
            envases = int(input("Ingrese el numero de envases nuevamente. Tenga en cuenta que el numero debe ser mayor o igual a 500: "))
        empresa.realizar_pedido_materia_prima('envases', envases, 500)
        
        etiquetas = int(input("Ingrese el numero de etiquetas. Tenga en cuenta que el numero debe ser mayor o igual a 5000: "))
        while etiquetas<5000:
            etiquetas = int(input("Ingrese el numero de etiquetas nuevamente. Tenga en cuenta que el numero debe ser mayor o igual a 5000: "))
        empresa.realizar_pedido_materia_prima('etiquetas', etiquetas, 100)
        
        melatonina_pura = float(input("Ingrese el numero de melatonina pura. Tenga en cuenta que el numero debe ser mayor o igual a 2000: "))
        while melatonina_pura<2000:
            melatonina_pura = float(input("Ingrese el numero de melatonina pura nuevamente. Tenga en cuenta que el numero debe ser mayor o igual a 2000: "))
        empresa.realizar_pedido_materia_prima('melatonina_pura', melatonina_pura, 800)
        
        estearato_magnesio = float(input("Ingrese el numero de estearato de magnesio. Tenga en cuenta que el numero debe ser mayor o igual a 3000: "))
        while estearato_magnesio<3000:
            estearato_magnesio = float(input("Ingrese el numero de estearato de magnesio nuevamente. Tenga en cuenta que el numero debe ser mayor o igual a 3000: "))
        empresa.realizar_pedido_materia_prima('estearato_magnesio', estearato_magnesio, 67)
        
        celulosa_microcristalina = float(input("Ingrese el numero de celulosa microcristalina. Tenga en cuenta que el numero debe ser mayor o igual a 20000: "))
        while celulosa_microcristalina<20000:
            celulosa_microcristalina = float(input("Ingrese el numero de celulosa microcristalina nuevamente. Tenga en cuenta que el numero debe ser mayor o igual a 20000: "))
        empresa.realizar_pedido_materia_prima('celulosa_microcristalina', celulosa_microcristalina, 50)
        
        
    elif seleccion == 2:
        # Revisar inventario
        empresa.revisar_inventario()
    elif seleccion == 3:
        # Convertir a productos de venta
        productos_convertir = int(input("Ingrese la cantidad de productos que desea convertir: "))
        empresa.convertir_a_venta(productos_convertir)
    elif seleccion == 4:
        # Calcular pedidos proyectados
        ventas_proyectadas = int(input("Ingrese el numero de ventas proyectadas por mes: "))
        pedidos_necesarios = empresa.calcular_pedidos_proyectados(ventas_proyectadas)
        # Muestra la cantidad de pedidos requeridos para las ventadas proyectadas
        print(f"Necesita la siguiente cantidad de productos para obtener {ventas_proyectadas} ventas, teniendo en cuenta el inventario de materia prima:\n ")
        for producto, cantidad in pedidos_necesarios.items():
            print(f"{producto}: {cantidad:,.2f}")
    else:
        print("Adios.")
        break
        


# In[ ]:




