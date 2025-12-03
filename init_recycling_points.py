"""
Initialize Recycling Points Database
Adds sample recycling points for different cities and categories
"""

from recycling_location import RecyclingLocationManager
from config import AVAILABLE_CITIES


def init_sample_points():
    """
    Initialize sample recycling points for demonstration.
    """
    manager = RecyclingLocationManager()
    
    print("Initializing recycling points database...")
    
    # Sample points for different cities and categories
    # Coordinates are approximate for major Chinese cities
    
    sample_points = [
        # Shanghai (Shanghai)
        {
            'name': '上海有害垃圾回收站（黄浦区）',
            'category': 'Hazardous',
            'latitude': 31.2304,
            'longitude': 121.4737,
            'address': '上海市黄浦区南京东路100号',
            'city_id': 'shanghai',
            'description': '专门回收电池、荧光灯管等有害垃圾',
            'phone': '021-12345678',
            'opening_hours': '周一至周日 9:00-18:00'
        },
        {
            'name': '上海可回收物回收点（浦东新区）',
            'category': 'Recyclable',
            'latitude': 31.2304,
            'longitude': 121.5000,
            'address': '上海市浦东新区陆家嘴环路1000号',
            'city_id': 'shanghai',
            'description': '回收纸张、塑料、金属、玻璃等可回收物',
            'phone': '021-12345679',
            'opening_hours': '周一至周日 8:00-20:00'
        },
        {
            'name': '上海厨余垃圾处理站（徐汇区）',
            'category': 'Kitchen',
            'latitude': 31.1900,
            'longitude': 121.4400,
            'address': '上海市徐汇区淮海中路1000号',
            'city_id': 'shanghai',
            'description': '专门处理厨余垃圾',
            'phone': '021-12345680',
            'opening_hours': '周一至周日 6:00-22:00'
        },
        
        # Beijing (Beijing)
        {
            'name': '北京有害垃圾回收站（朝阳区）',
            'category': 'Hazardous',
            'latitude': 39.9042,
            'longitude': 116.4074,
            'address': '北京市朝阳区建国门外大街1号',
            'city_id': 'beijing',
            'description': '专门回收电池、荧光灯管等有害垃圾',
            'phone': '010-12345678',
            'opening_hours': '周一至周日 9:00-18:00'
        },
        {
            'name': '北京可回收物回收点（海淀区）',
            'category': 'Recyclable',
            'latitude': 39.9593,
            'longitude': 116.2980,
            'address': '北京市海淀区中关村大街1号',
            'city_id': 'beijing',
            'description': '回收纸张、塑料、金属、玻璃等可回收物',
            'phone': '010-12345679',
            'opening_hours': '周一至周日 8:00-20:00'
        },
        {
            'name': '北京厨余垃圾处理站（西城区）',
            'category': 'Kitchen',
            'latitude': 39.9139,
            'longitude': 116.3668,
            'address': '北京市西城区西单北大街1号',
            'city_id': 'beijing',
            'description': '专门处理厨余垃圾',
            'phone': '010-12345680',
            'opening_hours': '周一至周日 6:00-22:00'
        },
        
        # Shenzhen (Shenzhen)
        {
            'name': '深圳有害垃圾回收站（南山区）',
            'category': 'Hazardous',
            'latitude': 22.5431,
            'longitude': 114.0579,
            'address': '深圳市南山区科技园南区',
            'city_id': 'shenzhen',
            'description': '专门回收电池、荧光灯管等有害垃圾',
            'phone': '0755-12345678',
            'opening_hours': '周一至周日 9:00-18:00'
        },
        {
            'name': '深圳可回收物回收点（福田区）',
            'category': 'Recyclable',
            'latitude': 22.5234,
            'longitude': 114.0579,
            'address': '深圳市福田区中心区',
            'city_id': 'shenzhen',
            'description': '回收纸张、塑料、金属、玻璃等可回收物',
            'phone': '0755-12345679',
            'opening_hours': '周一至周日 8:00-20:00'
        },
        
        # Guangzhou (Guangzhou)
        {
            'name': '广州有害垃圾回收站（天河区）',
            'category': 'Hazardous',
            'latitude': 23.1291,
            'longitude': 113.2644,
            'address': '广州市天河区天河路123号',
            'city_id': 'guangzhou',
            'description': '专门回收电池、荧光灯管等有害垃圾',
            'phone': '020-12345678',
            'opening_hours': '周一至周日 9:00-18:00'
        },
        {
            'name': '广州可回收物回收点（越秀区）',
            'category': 'Recyclable',
            'latitude': 23.1291,
            'longitude': 113.2644,
            'address': '广州市越秀区北京路1号',
            'city_id': 'guangzhou',
            'description': '回收纸张、塑料、金属、玻璃等可回收物',
            'phone': '020-12345679',
            'opening_hours': '周一至周日 8:00-20:00'
        },
        
        # Default city (general locations)
        {
            'name': '通用有害垃圾回收站',
            'category': 'Hazardous',
            'latitude': 31.2304,
            'longitude': 121.4737,
            'address': '请查询当地环保部门获取具体地址',
            'city_id': 'default',
            'description': '专门回收电池、荧光灯管等有害垃圾',
            'phone': '400-123-4567',
            'opening_hours': '周一至周日 9:00-18:00'
        },
        {
            'name': '通用可回收物回收点',
            'category': 'Recyclable',
            'latitude': 31.2304,
            'longitude': 121.5000,
            'address': '请查询当地环保部门获取具体地址',
            'city_id': 'default',
            'description': '回收纸张、塑料、金属、玻璃等可回收物',
            'phone': '400-123-4568',
            'opening_hours': '周一至周日 8:00-20:00'
        },
        {
            'name': '通用厨余垃圾处理站',
            'category': 'Kitchen',
            'latitude': 31.1900,
            'longitude': 121.4400,
            'address': '请查询当地环保部门获取具体地址',
            'city_id': 'default',
            'description': '专门处理厨余垃圾',
            'phone': '400-123-4569',
            'opening_hours': '周一至周日 6:00-22:00'
        },
        {
            'name': '通用其他垃圾处理站',
            'category': 'Other',
            'latitude': 31.2000,
            'longitude': 121.4500,
            'address': '请查询当地环保部门获取具体地址',
            'city_id': 'default',
            'description': '处理其他垃圾',
            'phone': '400-123-4570',
            'opening_hours': '周一至周日 6:00-22:00'
        }
    ]
    
    # Add all sample points
    added_count = 0
    for point in sample_points:
        try:
            manager.add_recycling_point(**point)
            added_count += 1
            print(f"✓ Added: {point['name']} ({point['city_id']}, {point['category']})")
        except Exception as e:
            print(f"✗ Failed to add {point['name']}: {e}")
    
    print(f"\n✓ Successfully added {added_count} recycling points")
    print(f"✓ Database initialized at: {manager.db_path}")


if __name__ == '__main__':
    init_sample_points()

