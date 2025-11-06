using UnityEngine;
using UnityEngine.SceneManagement;

public class SceneLoader : MonoBehaviour
{
    void Update()
    {
        // Check if a key from 1 to 0 is pressed
        if (Input.GetKeyDown(KeyCode.Alpha1))
        {
            LoadSceneByNumber(0);
        }
        else if (Input.GetKeyDown(KeyCode.Alpha2))
        {
            LoadSceneByNumber(1);
        }
        else if (Input.GetKeyDown(KeyCode.Alpha3))
        {
            LoadSceneByNumber(2);
        }
        else if (Input.GetKeyDown(KeyCode.Alpha4))
        {
            LoadSceneByNumber(3);
        }
        else if (Input.GetKeyDown(KeyCode.Alpha5))
        {
            LoadSceneByNumber(4);
        }
        else if (Input.GetKeyDown(KeyCode.Alpha6))
        {
            LoadSceneByNumber(5);
        }
        else if (Input.GetKeyDown(KeyCode.Alpha7))
        {
            LoadSceneByNumber(6);
        }
        else if (Input.GetKeyDown(KeyCode.Alpha8))
        {
            LoadSceneByNumber(7);
        }
        else if (Input.GetKeyDown(KeyCode.Alpha9))
        {
            LoadSceneByNumber(10);
        }
        else if (Input.GetKeyDown(KeyCode.Alpha0))
        {
            LoadSceneByNumber(11);
        }
    }

    private void LoadSceneByNumber(int sceneNumber)
    {
        // Assuming you have your scenes in the build settings in order from 0 to 9
        if (sceneNumber >= 0 && sceneNumber < SceneManager.sceneCountInBuildSettings)
        {
            SceneManager.LoadScene(sceneNumber);
        }
        else
        {
            Debug.LogError("Scene number " + sceneNumber + " is not available in the build.");
        }
    }
    public void LoadStoneAge()
    {
        SceneManager.LoadScene(0);
    }

    public void LoadMiddleAge()
    {
        SceneManager.LoadScene(4);
    }

    public void LoadSpaceAge()
    {
        SceneManager.LoadScene(9);
    }
 }
