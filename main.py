import asyncio
import random
import time
from argparse import ArgumentParser

from playground import playgrounds
from sport_api import FudanAPI, get_routes

# 主函数，异步执行
async def main():
    parser = ArgumentParser()
    parser.add_argument('-v', '--view', action='store_true', help="list available routes")  # 列出可用的路线
    parser.add_argument('-r', '--route', help="set route ID", type=int)  # 设置路线ID
    parser.add_argument('-t', '--time', help="total time, in seconds", type=int)  # 设置总时间（秒）
    parser.add_argument('-d', '--distance', help="total distance, in meters", type=int)  # 设置总距离（米）
    parser.add_argument('-q', '--delay', action='store_true', help="delay for random time")  # 随机延迟
    args = parser.parse_args()

    if args.view:
        routes = await get_routes()  # 获取路线
        supported_routes = filter(lambda r: r.id in playgrounds, routes)  # 过滤支持的路线
        for route in supported_routes:
            route.pretty_print()  # 打印路线信息
        exit()

    if args.route:
        # 设置距离
        distance = 1200
        if args.distance:
            distance = args.distance
        distance += random.uniform(-5.0, 25.0)  # 随机调整距离

        # 设置时间
        total_time = 360
        if args.time:
            total_time = args.time
        total_time += random.uniform(-10.0, 10.0)  # 随机调整时间

        # 从服务器获取路线
        routes = await get_routes()
        for route in routes:
            if route.id == args.route:
                selected_route = route
                break
        else:
            raise ValueError(f'不存在id为{args.route}的route')  # 抛出异常

        # 随机延迟，用于GitHub Action部署
        if args.delay:
            sleep_time = random.randint(0, 240)
            time.sleep(sleep_time)

        # 准备并开始运行
        automator = FudanAPI(selected_route)
        playground = playgrounds[args.route]
        current_distance = 0
        await automator.start()  # 开始
        print(f"START: {selected_route.name}")
        while current_distance < distance:
            current_distance += distance / total_time  # 更新当前距离
            message, _ = await asyncio.gather(
                automator.update(playground.random_offset(current_distance)), asyncio.sleep(1))  # 更新状态
            print(f"UPDATE: {message} ({current_distance}m / {distance}m)")
        finish_message = await automator.finish(playground.coordinate(distance))  # 完成
        print(f"FINISHED: {finish_message}")

if __name__ == '__main__':
    asyncio.run(main())  # 运行主函数
