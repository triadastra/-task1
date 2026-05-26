#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BRK-B 数据展示中心 - 本地服务器
用法: python brk_b_dashboard/start_server.py
"""
import http.server
import socketserver
import os
import webbrowser
import socket
from threading import Timer

PORT = 8080

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

def find_free_port(start=8080):
    port = start
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
        port += 1

# 切换到项目根目录（本文件的父目录）
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PORT = find_free_port(PORT)

with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
    print(f"\n{'='*50}")
    print(f"  BRK-B 数据展示中心 已启动")
    print(f"{'='*50}")
    print(f"\n  本地地址: http://localhost:{PORT}/brk_b_dashboard/")
    print(f"  网络地址: http://{socket.gethostbyname(socket.gethostname())}:{PORT}/brk_b_dashboard/")
    print(f"\n  按 Ctrl+C 停止服务器")
    print(f"{'='*50}\n")
    
    # 自动打开浏览器
    Timer(1.0, lambda: webbrowser.open(f'http://localhost:{PORT}/brk_b_dashboard/')).start()
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n服务器已停止。")
