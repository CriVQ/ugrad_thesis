using System;
using System.IO.Ports;
using System.Threading;
using UnityEngine;

public class ArduinoCom : MonoBehaviour
{
    private SerialPort _port;
    public string portName = "COM5";  // Ensure this is the correct port
    public int _baudRate = 115200;
    
    private Thread _threadEMG;
    private bool _isThreadEMGRunning = false;
    private float lastValue;
    private float value_to_send;

    public void Start()
    {
        if (_port == null) // Only create the port if it doesn't exist
        {
            _port = new SerialPort(portName, _baudRate)
            {
                ReadTimeout = 100, // Prevent Unity from freezing
                WriteTimeout = 100
            };
        }

        OpenPort();

        if (_port.IsOpen && !_isThreadEMGRunning)
        {
            _isThreadEMGRunning = true;
            _threadEMG = new Thread(ReadData);
            _threadEMG.Start();
            Debug.Log("Receiving data...");
        }
    }

    public void OpenPort()
    {
        if (_port != null && !_port.IsOpen)
        {
            try
            {
                _port.Open();
                Debug.Log("Connected to " + portName);
            }
            catch (Exception e)
            {
                Debug.LogError("Error opening port: " + e);
            }
        }
    }

    public void Stop()
    {
        _isThreadEMGRunning = false;

        if (_threadEMG != null && _threadEMG.IsAlive)
        {
            _threadEMG.Join(); // Wait for thread to finish safely
        }

        if (_port != null && _port.IsOpen)
        {
            _port.Close();
            Debug.Log("Port closed.");
        }
    }

    private void ReadData()
    {
        while (_isThreadEMGRunning && _port != null && _port.IsOpen)
        {
            try
            {
                if (_port.BytesToRead > 0) // Ensure there's data before reading
                {
                    string data = _port.ReadLine().Trim();
                    if (!string.IsNullOrEmpty(data))
                    {
                        float.TryParse(data, out lastValue);
                        value_to_send = lastValue;
                    }
                }
            }
            catch (TimeoutException)
            {
                // Ignore timeouts to prevent Unity freezing
            }
            catch (Exception e)
            {
                Debug.LogError("Serial read error: " + e);
            }
        }
    }

    public float GetData()
    {
        return value_to_send;
    }

    public void SendData(string data)
    {
        try
        {
            if (_port != null && _port.IsOpen)
            {
                _port.WriteLine(data);
            }
        }
        catch (Exception e)
        {
            Debug.LogError("Error sending data: " + e);
        }
    }

    private void OnApplicationQuit()
    {
        Stop();
    }
}