import asyncio
from tensorbeat.sarosh_gen import GenerateMusicResponse, SaroshGeneratorBase
from grpclib.server import Server


class SaroshGeneratorService(SaroshGeneratorBase):

    async def generate_music(self, yt_playlist_url: str) -> GenerateMusicResponse:
        resp = GenerateMusicResponse()
        resp.song = bytes('Hello World')
        return resp


async def start_server():
    HOST = "127.0.0.1"
    PORT = 1337
    server = Server([SaroshGeneratorService()])
    await server.start(HOST, PORT)
    await server.serve_forever()

if __name__ == '__main__':
    asyncio.run(start_server())