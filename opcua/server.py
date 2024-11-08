
import asyncio
from asyncua import ua, uamethod, Server
from asyncua.common.instantiate_util import instantiate
class service_server:

    def __init__(self):
        self.server = None
        self.custom_data_types = {"Name":[], "Class":[]}

    async def main(self):
        print("start server")
        common_model = "common.xml"
        cutting_model = "opt-assign.xml"
        #start the server
        self.server = Server()
        await self.server.init()
        server_address = "127.0.0.1"
        port = 3001
        self.server.set_endpoint("opc.tcp://"+str(server_address)+":"+str(port))
        #load the information models
        await self.server.import_xml(common_model)
        #load the custom data types from the common model
        custom_objs_common_model = await self.server.load_data_type_definitions()
        await self.server.import_xml(cutting_model)
        # load the custom data types from the common model
        custom_objs_service_model = await self.server.load_data_type_definitions()
        uri = "Example_Server_Opt_Assign "
        self.idx = await self.server.register_namespace(uri)



        ###################
        # Import custom data types
        ###################
        # create a dictionary that contains tha names of the custom data type definitions and corresponding classes
        # custom_objects = {"Name":[], "Class":[]}
        # common namespace
        await self.append_custom_type_dict(custom_objs_common_model)
        # second namespace
        await self.append_custom_type_dict(custom_objs_service_model)

        ###############
        #declare the instance of the module type
        ###############
        #get the node of the data type
        module_type_node = self.server.get_node("ns=3;i=15024")
        module_type_object = "OptAssignModuleType"
        await self.server.nodes.objects.add_object(3, str(module_type_object), module_type_node)

        async with self.server:
            while True:
                await asyncio.sleep(1)


    async def append_custom_type_dict(self, custom_type_definitions):
            for name, obj in custom_type_definitions.items():
                self.custom_data_types["Name"].append(name)
                self.custom_data_types["Class"].append(obj)

    async def get_struct_data_type(self, data_type, value):
        for i in range(len(self.custom_data_types["Name"])):
            if str(self.custom_data_types["Name"][i]) == str(data_type):
                result_data = self.custom_data_types["Class"][i]() if value ==None else self.custom_data_types["Class"][i](value)
                return result_data

if __name__ == "__main__":

    asyncio.run(service_server().main())
