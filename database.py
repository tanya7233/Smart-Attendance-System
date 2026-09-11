"""
Database Module for Smart Attendance System
Handles database operations, attendance records, and report generation

Author: AI Assistant
Date: 2025
"""

import sqlite3
import pandas as pd
import os
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Tuple
import logging
import csv
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = "data/attendance.db"):
        """
        Initialize the Database Manager
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self.ensure_data_directory()
    
    def ensure_data_directory(self) -> None:
        """Ensure the data directory exists"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def initialize_database(self) -> None:
        """Initialize the database with required tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create students table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    email TEXT,
                    phone TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create attendance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_name TEXT NOT NULL,
                    date DATE NOT NULL,
                    time TIME NOT NULL,
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(student_name, date)
                )
            ''')
            
            # Create attendance_log table for detailed logging
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attendance_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_name TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    confidence REAL,
                    status TEXT DEFAULT 'present'
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def add_student(self, name: str, email: str = None, phone: str = None) -> bool:
        """
        Add a new student to the database
        
        Args:
            name (str): Student name
            email (str): Student email (optional)
            phone (str): Student phone (optional)
            
        Returns:
            bool: Success status
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO students (name, email, phone)
                VALUES (?, ?, ?)
            ''', (name, email, phone))
            
            conn.commit()
            conn.close()
            logger.info(f"Added student: {name}")
            return True
            
        except sqlite3.IntegrityError:
            logger.warning(f"Student {name} already exists")
            return False

    def delete_student(self, name: str) -> bool:
        """
        Delete a student and related attendance records
        
        Args:
            name (str): Student name
        
        Returns:
            bool: Success status
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Delete related attendance and logs first
            cursor.execute('DELETE FROM attendance WHERE student_name = ?', (name,))
            cursor.execute('DELETE FROM attendance_log WHERE student_name = ?', (name,))
            # Delete the student
            cursor.execute('DELETE FROM students WHERE name = ?', (name,))

            conn.commit()
            deleted = cursor.rowcount
            conn.close()

            if deleted > 0:
                logger.info(f"Deleted student and related data: {name}")
                return True
            else:
                logger.warning(f"Student not found for deletion: {name}")
                return False
        except Exception as e:
            logger.error(f"Error deleting student {name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error adding student {name}: {e}")
            return False
    
    def mark_attendance(self, student_name: str, confidence: float = 0.0, target_date: str = None, target_time: str = None) -> bool:
        """
        Mark attendance for a student
        
        Args:
            student_name (str): Name of the student
            confidence (float): Recognition confidence score
            
        Returns:
            bool: Success status (True if new attendance, False if already marked)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Store as strings to avoid sqlite binding issues
            current_date = target_date if target_date else date.today().isoformat()
            current_time = target_time if target_time else datetime.now().strftime('%H:%M:%S')
            
            # Check if attendance already marked today
            cursor.execute('''
                SELECT id FROM attendance 
                WHERE student_name = ? AND date = ?
            ''', (student_name, current_date))
            
            if cursor.fetchone():
                conn.close()
                logger.info(f"Attendance already marked for {student_name} today")
                return False
            
            # Mark attendance
            cursor.execute('''
                INSERT INTO attendance (student_name, date, time, confidence)
                VALUES (?, ?, ?, ?)
            ''', (student_name, current_date, current_time, confidence))
            
            # Log the attendance
            cursor.execute('''
                INSERT INTO attendance_log (student_name, confidence)
                VALUES (?, ?)
            ''', (student_name, confidence))
            
            conn.commit()
            conn.close()
            logger.info(f"Attendance marked for {student_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error marking attendance for {student_name}: {e}")
            return False
    
    def get_daily_attendance(self, target_date: date = None) -> List[Dict]:
        """
        Get attendance for a specific date
        
        Args:
            target_date (date): Target date (default: today)
            
        Returns:
            List[Dict]: List of attendance records
        """
        if target_date is None:
            target_date = date.today()
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT student_name, date, time, confidence
                FROM attendance
                WHERE date = ?
                ORDER BY time ASC
            ''', (target_date,))
            
            rows = cursor.fetchall()
            conn.close()
            
            attendance_list = []
            for row in rows:
                attendance_list.append({
                    'student_name': row[0],
                    'date': row[1],
                    'time': row[2],
                    'confidence': row[3]
                })
            
            return attendance_list
            
        except Exception as e:
            logger.error(f"Error getting daily attendance: {e}")
            return []
    
    def get_monthly_attendance(self, year: int, month: int) -> List[Dict]:
        """
        Get attendance for a specific month
        
        Args:
            year (int): Year
            month (int): Month
            
        Returns:
            List[Dict]: List of attendance records
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT student_name, date, time, confidence
                FROM attendance
                WHERE strftime('%Y', date) = ? AND strftime('%m', date) = ?
                ORDER BY date ASC, time ASC
            ''', (str(year), f"{month:02d}"))
            
            rows = cursor.fetchall()
            conn.close()
            
            attendance_list = []
            for row in rows:
                attendance_list.append({
                    'student_name': row[0],
                    'date': row[1],
                    'time': row[2],
                    'confidence': row[3]
                })
            
            return attendance_list
            
        except Exception as e:
            logger.error(f"Error getting monthly attendance: {e}")
            return []
    
    def get_student_attendance_summary(self, student_name: str, start_date: date = None, end_date: date = None) -> Dict:
        """
        Get attendance summary for a specific student
        
        Args:
            student_name (str): Name of the student
            start_date (date): Start date (optional)
            end_date (date): End date (optional)
            
        Returns:
            Dict: Attendance summary
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = '''
                SELECT COUNT(*) as total_days
                FROM attendance
                WHERE student_name = ?
            '''
            params = [student_name]
            
            if start_date and end_date:
                query += ' AND date BETWEEN ? AND ?'
                params.extend([start_date, end_date])
            
            cursor.execute(query, params)
            result = cursor.fetchone()
            total_present = result[0] if result else 0
            
            # Calculate total working days (assuming Monday to Friday)
            if start_date and end_date:
                total_working_days = self.calculate_working_days(start_date, end_date)
            else:
                # Default to current month but not before student's creation date
                today = date.today()
                start_of_month = today.replace(day=1)
                # Fetch student's created_at date
                try:
                    conn2 = sqlite3.connect(self.db_path)
                    cur2 = conn2.cursor()
                    cur2.execute('SELECT DATE(created_at) FROM students WHERE name = ?', (student_name,))
                    row = cur2.fetchone()
                    conn2.close()
                    created_date = date.fromisoformat(row[0]) if row and row[0] else start_of_month
                except Exception:
                    created_date = start_of_month
                effective_start = max(start_of_month, created_date)
                total_working_days = self.calculate_working_days(effective_start, today)
            
            total_absent = max(0, total_working_days - total_present)
            attendance_percentage = (total_present / total_working_days * 100) if total_working_days > 0 else 0
            
            conn.close()
            
            return {
                'student_name': student_name,
                'total_present': total_present,
                'total_absent': total_absent,
                'total_working_days': total_working_days,
                'attendance_percentage': round(attendance_percentage, 2)
            }
            
        except Exception as e:
            logger.error(f"Error getting attendance summary for {student_name}: {e}")
            return {}
    
    def calculate_working_days(self, start_date: date, end_date: date) -> int:
        """
        Calculate working days between two dates (excluding weekends)
        
        Args:
            start_date (date): Start date
            end_date (date): End date
            
        Returns:
            int: Number of working days
        """
        working_days = 0
        current_date = start_date
        
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Monday = 0, Friday = 4
                working_days += 1
            current_date += timedelta(days=1)
        
        return working_days
    
    def get_all_students(self) -> List[Dict]:
        """
        Get all students from the database
        
        Returns:
            List[Dict]: List of all students
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT name, email, phone, created_at FROM students ORDER BY name')
            rows = cursor.fetchall()
            conn.close()
            
            students = []
            for row in rows:
                students.append({
                    'name': row[0],
                    'email': row[1],
                    'phone': row[2],
                    'created_at': row[3]
                })
            
            return students
            
        except Exception as e:
            logger.error(f"Error getting all students: {e}")
            return []
    
    def export_to_csv(self, attendance_data: List[Dict], filename: str) -> bool:
        """
        Export attendance data to CSV file
        
        Args:
            attendance_data (List[Dict]): Attendance data to export
            filename (str): Output filename
            
        Returns:
            bool: Success status
        """
        try:
            filepath = os.path.join("data/reports", filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            if not attendance_data:
                logger.warning("No data to export")
                return False
            
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = attendance_data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for row in attendance_data:
                    writer.writerow(row)
            
            logger.info(f"Data exported to CSV: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False
    
    def export_to_excel(self, attendance_data: List[Dict], filename: str) -> bool:
        """
        Export attendance data to Excel file
        
        Args:
            attendance_data (List[Dict]): Attendance data to export
            filename (str): Output filename
            
        Returns:
            bool: Success status
        """
        try:
            filepath = os.path.join("data/reports", filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            if not attendance_data:
                logger.warning("No data to export")
                return False
            
            df = pd.DataFrame(attendance_data)
            df.to_excel(filepath, index=False, engine='openpyxl')
            
            logger.info(f"Data exported to Excel: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to Excel: {e}")
            return False
    
    def generate_pdf_report(self, attendance_data: List[Dict], summary_data: List[Dict], filename: str, title: str = "Attendance Report") -> bool:
        """
        Generate PDF report with attendance data and summary
        
        Args:
            attendance_data (List[Dict]): Attendance data
            summary_data (List[Dict]): Summary data
            filename (str): Output filename
            title (str): Report title
            
        Returns:
            bool: Success status
        """
        try:
            filepath = os.path.join("data/reports", filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            doc = SimpleDocTemplate(filepath, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=1  # Center alignment
            )
            story.append(Paragraph(title, title_style))
            story.append(Spacer(1, 20))
            
            # Generated date
            date_style = ParagraphStyle(
                'DateStyle',
                parent=styles['Normal'],
                fontSize=12,
                alignment=1
            )
            story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", date_style))
            story.append(Spacer(1, 30))
            
            # Summary section
            if summary_data:
                story.append(Paragraph("Attendance Summary", styles['Heading2']))
                story.append(Spacer(1, 10))
                
                summary_table_data = [['Student Name', 'Present Days', 'Absent Days', 'Total Days', 'Attendance %']]
                for summary in summary_data:
                    summary_table_data.append([
                        summary.get('student_name', ''),
                        str(summary.get('total_present', 0)),
                        str(summary.get('total_absent', 0)),
                        str(summary.get('total_working_days', 0)),
                        f"{summary.get('attendance_percentage', 0)}%"
                    ])
                
                summary_table = Table(summary_table_data)
                summary_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 14),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(summary_table)
                story.append(Spacer(1, 30))
            
            # Detailed attendance section
            if attendance_data:
                story.append(Paragraph("Detailed Attendance Records", styles['Heading2']))
                story.append(Spacer(1, 10))
                
                # Create table data
                table_data = [['Student Name', 'Date', 'Time', 'Confidence']]
                for record in attendance_data:
                    table_data.append([
                        record.get('student_name', ''),
                        record.get('date', ''),
                        record.get('time', ''),
                        f"{record.get('confidence', 0):.2f}" if record.get('confidence') else 'N/A'
                    ])
                
                # Create table
                table = Table(table_data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(table)
            
            # Build PDF
            doc.build(story)
            logger.info(f"PDF report generated: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            return False
    
    def get_attendance_statistics(self, start_date: date = None, end_date: date = None) -> Dict:
        """
        Get attendance statistics for a date range
        
        Args:
            start_date (date): Start date (optional)
            end_date (date): End date (optional)
            
        Returns:
            Dict: Attendance statistics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get total students
            total_students_query = "SELECT COUNT(DISTINCT name) FROM students"
            total_students = pd.read_sql_query(total_students_query, conn).iloc[0, 0]
            
            # Get attendance data for the date range
            attendance_query = '''
                SELECT student_name, date, COUNT(*) as attendance_count
                FROM attendance
            '''
            params = []
            
            if start_date and end_date:
                attendance_query += ' WHERE date BETWEEN ? AND ?'
                params.extend([start_date, end_date])
            
            attendance_query += ' GROUP BY student_name, date'
            
            df = pd.read_sql_query(attendance_query, conn, params=params)
            conn.close()
            
            if df.empty:
                return {
                    'total_students': total_students,
                    'students_present_today': 0,
                    'students_absent_today': total_students,
                    'average_attendance': 0.0,
                    'total_attendance_records': 0
                }
            
            # Calculate statistics
            unique_students_present = df['student_name'].nunique()
            total_records = len(df)
            
            # Today's attendance
            today = date.today()
            today_present = len(df[df['date'] == str(today)]) if 'date' in df.columns else 0
            today_absent = max(0, total_students - today_present)
            
            # Calculate working days for average
            if start_date and end_date:
                working_days = self.calculate_working_days(start_date, end_date)
            else:
                # Default to current month
                today = date.today()
                start_of_month = today.replace(day=1)
                working_days = self.calculate_working_days(start_of_month, today)
            
            average_attendance = (unique_students_present / total_students * 100) if total_students > 0 else 0
            
            return {
                'total_students': total_students,
                'students_present_today': today_present,
                'students_absent_today': today_absent,
                'average_attendance': round(average_attendance, 2),
                'total_attendance_records': total_records,
                'working_days_in_period': working_days
            }
            
        except Exception as e:
            logger.error(f"Error getting attendance statistics: {e}")
            return {}
    
    def cleanup_old_logs(self, days_to_keep: int = 30) -> None:
        """
        Clean up old attendance logs
        
        Args:
            days_to_keep (int): Number of days to keep logs for
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            cursor.execute('''
                DELETE FROM attendance_log 
                WHERE timestamp < ?
            ''', (cutoff_date,))
            
            deleted_rows = cursor.rowcount
            conn.commit()
            conn.close()
            
            logger.info(f"Cleaned up {deleted_rows} old log entries")
            
        except Exception as e:
            logger.error(f"Error cleaning up old logs: {e}")

    def reset_all_data(self) -> bool:
        """
        Delete all rows from students, attendance, and attendance_log tables.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM attendance_log')
            cursor.execute('DELETE FROM attendance')
            cursor.execute('DELETE FROM students')
            conn.commit()
            try:
                cursor.execute('VACUUM')
                conn.commit()
            except Exception:
                pass
            conn.close()
            logger.info("All database tables cleared")
            return True
        except Exception as e:
            logger.error(f"Error resetting database: {e}")
            return False

class ReportGenerator:
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize the Report Generator
        
        Args:
            db_manager (DatabaseManager): Database manager instance
        """
        self.db_manager = db_manager
    
    def generate_daily_report(self, target_date: date = None) -> Dict:
        """
        Generate daily attendance report
        
        Args:
            target_date (date): Target date (default: today)
            
        Returns:
            Dict: Report data
        """
        if target_date is None:
            target_date = date.today()
        
        # Get daily attendance
        attendance_data = self.db_manager.get_daily_attendance(target_date)
        
        # Get all students for summary
        all_students = self.db_manager.get_all_students()
        
        # Create summary
        summary_data = []
        present_students = {record['student_name'] for record in attendance_data}
        
        for student in all_students:
            is_present = student['name'] in present_students
            summary_data.append({
                'student_name': student['name'],
                'total_present': 1 if is_present else 0,
                'total_absent': 0 if is_present else 1,
                'total_working_days': 1,
                'attendance_percentage': 100.0 if is_present else 0.0
            })
        
        return {
            'attendance_data': attendance_data,
            'summary_data': summary_data,
            'report_date': target_date,
            'report_type': 'daily'
        }
    
    def generate_monthly_report(self, year: int, month: int) -> Dict:
        """
        Generate monthly attendance report
        
        Args:
            year (int): Year
            month (int): Month
            
        Returns:
            Dict: Report data
        """
        # Get monthly attendance
        attendance_data = self.db_manager.get_monthly_attendance(year, month)
        
        # Get all students for summary
        all_students = self.db_manager.get_all_students()
        
        # Calculate date range
        start_date = date(year, month, 1)
        if month == 12:
            end_date = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = date(year, month + 1, 1) - timedelta(days=1)
        
        # Create summary
        summary_data = []
        for student in all_students:
            summary = self.db_manager.get_student_attendance_summary(
                student['name'], start_date, end_date
            )
            if summary:
                summary_data.append(summary)
        
        return {
            'attendance_data': attendance_data,
            'summary_data': summary_data,
            'report_period': f"{year}-{month:02d}",
            'report_type': 'monthly'
        }
    
    def create_attendance_chart(self, summary_data: List[Dict]) -> str:
        """
        Create attendance chart and return as base64 encoded string
        
        Args:
            summary_data (List[Dict]): Summary data for chart
            
        Returns:
            str: Base64 encoded chart image
        """
        try:
            if not summary_data:
                return ""
            
            # Prepare data
            names = [item['student_name'] for item in summary_data]
            percentages = [item['attendance_percentage'] for item in summary_data]
            
            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Create bar chart
            bars = plt.bar(names, percentages, color='skyblue', edgecolor='navy', alpha=0.7)
            
            # Customize chart
            plt.title('Attendance Percentage by Student', fontsize=16, fontweight='bold')
            plt.xlabel('Students', fontsize=12)
            plt.ylabel('Attendance Percentage (%)', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 100)
            
            # Add value labels on bars
            for bar, percentage in zip(bars, percentages):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{percentage:.1f}%', ha='center', va='bottom')
            
            # Add grid
            plt.grid(axis='y', alpha=0.3)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save to bytes
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            
            # Convert to base64
            chart_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            plt.close()
            buffer.close()
            
            return chart_base64
            
        except Exception as e:
            logger.error(f"Error creating attendance chart: {e}")
            return ""