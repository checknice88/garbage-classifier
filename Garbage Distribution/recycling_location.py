"""
Recycling Location Management System
Manages recycling points and collection bins using SQLite database
"""

import sqlite3
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import os
from config import RECYCLING_LOCATION_DB_PATH
from map_service import MapService


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth using Haversine formula.
    
    Args:
        lat1, lon1: Latitude and longitude of first point in degrees
        lat2, lon2: Latitude and longitude of second point in degrees
        
    Returns:
        Distance in kilometers
    """
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    
    # Earth radius in kilometers
    R = 6371.0
    
    return R * c


class RecyclingLocationManager:
    """
    Manages recycling points and collection bins.
    Uses SQLite database for data persistence.
    """
    
    def __init__(self, db_path: str = None, map_service: MapService = None):
        """
        Initialize the recycling location manager.
        
        Args:
            db_path: Path to SQLite database file (default: from config)
            map_service: MapService instance for API searches (optional)
        """
        self.db_path = db_path or RECYCLING_LOCATION_DB_PATH
        self.map_service = map_service or MapService()
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Recycling points table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recycling_points (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                address TEXT,
                city_id TEXT DEFAULT 'default',
                description TEXT,
                phone TEXT,
                opening_hours TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        ''')
        
        # Create index for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_category ON recycling_points(category)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_city ON recycling_points(city_id)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_location ON recycling_points(latitude, longitude)
        ''')
        
        conn.commit()
        conn.close()
    
    def add_recycling_point(self, 
                           name: str,
                           category: str,
                           latitude: float,
                           longitude: float,
                           address: str = None,
                           city_id: str = 'default',
                           description: str = None,
                           phone: str = None,
                           opening_hours: str = None) -> int:
        """
        Add a new recycling point to the database.
        
        Args:
            name: Name of the recycling point
            category: Waste category (Recyclable, Hazardous, Kitchen, Other)
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            address: Physical address
            city_id: City identifier
            description: Additional description
            phone: Contact phone number
            opening_hours: Opening hours information
            
        Returns:
            ID of the newly created recycling point
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        cursor.execute('''
            INSERT INTO recycling_points 
            (name, category, latitude, longitude, address, city_id, 
             description, phone, opening_hours, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (name, category, latitude, longitude, address, city_id,
              description, phone, opening_hours, now, now))
        
        point_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return point_id
    
    def get_nearby_points(self,
                         latitude: float,
                         longitude: float,
                         category: str = None,
                         city_id: str = None,
                         radius_km: float = 5.0,
                         limit: int = 10) -> List[Dict]:
        """
        Find nearby recycling points within a specified radius.
        
        Args:
            latitude: User's latitude
            longitude: User's longitude
            category: Filter by category (optional)
            city_id: Filter by city (optional)
            radius_km: Search radius in kilometers
            limit: Maximum number of results
            
        Returns:
            List of dictionaries containing point information with distance
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build query
        query = 'SELECT id, name, category, latitude, longitude, address, city_id, description, phone, opening_hours FROM recycling_points WHERE 1=1'
        params = []
        
        if category:
            query += ' AND category = ?'
            params.append(category)
        
        if city_id:
            query += ' AND city_id = ?'
            params.append(city_id)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        # Calculate distances and filter by radius
        nearby_points = []
        for row in rows:
            point_lat = row[3]
            point_lon = row[4]
            distance = haversine_distance(latitude, longitude, point_lat, point_lon)
            
            if distance <= radius_km:
                point_dict = {
                    'id': row[0],
                    'name': row[1],
                    'category': row[2],
                    'latitude': point_lat,
                    'longitude': point_lon,
                    'address': row[5],
                    'city_id': row[6],
                    'description': row[7],
                    'phone': row[8],
                    'opening_hours': row[9],
                    'distance_km': round(distance, 2)
                }
                nearby_points.append(point_dict)
        
        # Sort by distance and limit results
        nearby_points.sort(key=lambda x: x['distance_km'])
        return nearby_points[:limit]
    
    def search_hybrid(self,
                     latitude: float,
                     longitude: float,
                     category: str = None,
                     city_id: str = None,
                     radius_km: float = 5.0,
                     limit: int = 10,
                     use_api: bool = True) -> Dict:
        """
        Hybrid search: Use API first, then supplement with database.
        
        Args:
            latitude: User's latitude
            longitude: User's longitude
            category: Filter by category (optional)
            city_id: Filter by city (optional)
            radius_km: Search radius in kilometers
            limit: Maximum number of results
            use_api: Whether to use API search (default: True)
            
        Returns:
            Dictionary with 'points' (list), 'source' (str), 'search_radius' (float)
        """
        all_points = []
        api_points = []
        db_points = []
        actual_radius = radius_km
        
        # Step 1: Try API search first (if enabled and category provided)
        if use_api and category:
            try:
                api_points = self.map_service.search_nearby_pois(
                    latitude=latitude,
                    longitude=longitude,
                    category=category,
                    radius_km=radius_km
                )
                
                # Convert API points to standard format
                for poi in api_points:
                    poi['source'] = 'gaode_api'
                    all_points.append(poi)
                
                # Update actual search radius based on API results
                if api_points:
                    max_distance = max(p.get('distance_km', 0) for p in api_points)
                    actual_radius = max(actual_radius, max_distance)
                    
            except Exception as e:
                print(f"API search error: {e}")
                api_points = []
        
        # Step 2: Supplement with database points
        try:
            db_points = self.get_nearby_points(
                latitude=latitude,
                longitude=longitude,
                category=category,
                city_id=city_id,
                radius_km=radius_km * 1.5,  # Slightly larger radius for DB
                limit=limit * 2  # Get more to filter duplicates
            )
            
            # Mark database points and add if not duplicate
            seen_locations = set()
            for poi in all_points:
                key = (round(poi['latitude'], 4), round(poi['longitude'], 4))
                seen_locations.add(key)
            
            for db_point in db_points:
                key = (round(db_point['latitude'], 4), round(db_point['longitude'], 4))
                if key not in seen_locations:
                    db_point['source'] = 'database'
                    all_points.append(db_point)
                    seen_locations.add(key)
                    
        except Exception as e:
            print(f"Database search error: {e}")
            db_points = []
        
        # Sort by distance and limit
        all_points.sort(key=lambda x: x.get('distance_km', float('inf')))
        all_points = all_points[:limit]
        
        # Determine primary source
        if api_points:
            source = 'api_with_db_supplement' if db_points else 'api_only'
        elif db_points:
            source = 'database_only'
        else:
            source = 'none'
        
        return {
            'points': all_points,
            'source': source,
            'search_radius': actual_radius,
            'api_count': len(api_points),
            'db_count': len([p for p in all_points if p.get('source') == 'database'])
        }
    
    def get_points_by_category(self, category: str, city_id: str = None) -> List[Dict]:
        """
        Get all recycling points for a specific category.
        
        Args:
            category: Waste category
            city_id: Filter by city (optional)
            
        Returns:
            List of recycling point dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if city_id:
            cursor.execute('''
                SELECT id, name, category, latitude, longitude, address, city_id, 
                       description, phone, opening_hours
                FROM recycling_points
                WHERE category = ? AND city_id = ?
            ''', (category, city_id))
        else:
            cursor.execute('''
                SELECT id, name, category, latitude, longitude, address, city_id, 
                       description, phone, opening_hours
                FROM recycling_points
                WHERE category = ?
            ''', (category,))
        
        rows = cursor.fetchall()
        conn.close()
        
        points = []
        for row in rows:
            point_dict = {
                'id': row[0],
                'name': row[1],
                'category': row[2],
                'latitude': row[3],
                'longitude': row[4],
                'address': row[5],
                'city_id': row[6],
                'description': row[7],
                'phone': row[8],
                'opening_hours': row[9]
            }
            points.append(point_dict)
        
        return points
    
    def find_nearest_point(self,
                          latitude: float,
                          longitude: float,
                          category: str = None,
                          city_id: str = None) -> Optional[Dict]:
        """
        Find the nearest recycling point.
        
        Args:
            latitude: User's latitude
            longitude: User's longitude
            category: Filter by category (optional)
            city_id: Filter by city (optional)
            
        Returns:
            Dictionary containing the nearest point information, or None if not found
        """
        nearby = self.get_nearby_points(latitude, longitude, category, city_id, radius_km=50.0, limit=1)
        return nearby[0] if nearby else None
    
    def get_all_points(self, city_id: str = None) -> List[Dict]:
        """
        Get all recycling points.
        
        Args:
            city_id: Filter by city (optional)
            
        Returns:
            List of all recycling point dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if city_id:
            cursor.execute('''
                SELECT id, name, category, latitude, longitude, address, city_id, 
                       description, phone, opening_hours
                FROM recycling_points
                WHERE city_id = ?
            ''', (city_id,))
        else:
            cursor.execute('''
                SELECT id, name, category, latitude, longitude, address, city_id, 
                       description, phone, opening_hours
                FROM recycling_points
            ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        points = []
        for row in rows:
            point_dict = {
                'id': row[0],
                'name': row[1],
                'category': row[2],
                'latitude': row[3],
                'longitude': row[4],
                'address': row[5],
                'city_id': row[6],
                'description': row[7],
                'phone': row[8],
                'opening_hours': row[9]
            }
            points.append(point_dict)
        
        return points
    
    def delete_point(self, point_id: int) -> bool:
        """
        Delete a recycling point by ID.
        
        Args:
            point_id: ID of the point to delete
            
        Returns:
            True if deleted, False if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM recycling_points WHERE id = ?', (point_id,))
        deleted = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        
        return deleted

